"""Per-period memory and efficiency tracker  (training/memory_tracker.py)

Logs the following metrics for each training period, for both INCA and baselines:

  peak_train_mb        – peak GPU/MPS memory allocated during trainer.train() [MB]
  infer_mb             – model static memory on a single dummy forward pass [MB]
  param_total          – total model parameter count at period end
  param_total_start    – total model parameter count at very first period start (base model)
  param_trainable      – trainable parameter count during this period
  param_delta          – trainable params added vs previous period (0 for period 0)
  cumulative_param_delta – total params added since period 0 (running sum of param_delta)
  param_growth_pct     – (param_total - param_total_start) / param_total_start × 100
  wall_time_s          – training wall time in seconds for this period
  acc_per_mb           – accuracy improvement ÷ peak_train_mb (computed on save)

Output: results/<run_id>/memory_log.json  (list of per-period dicts, one per period)

LLaMA-Pro parameter reference (for comparison):
  Base model:   LLaMA2-7B      ≈ 6 738M params
  Expanded:     LLaMA-Pro-8.3B ≈ 8 303M params
  Params added: ≈ 1 565M  (+23.2%)  — all 8 blocks, fixed schedule, pre-training only
  Per block:    ≈  196M  (8 blocks interleaved in 32-layer LLaMA2-7B)

INCA (FLAN-T5-large) reference:
  Base model:   FLAN-T5-large  ≈   783M params  (24 enc + 24 dec, D=1024)
  Per enc block: ≈  40–50M params  (4 encoder layers, D=1024, FFN=2816)
  Max new params: ≈ 240–300M  (up to 6 enc blocks, +31–38%) — adaptive, not fixed

Usage (in trainer code)
-----------------------
    from training.memory_tracker import MemoryTracker

    tracker = MemoryTracker(device=device, method="inca")

    # At the start of each period:
    tracker.period_start(period_id, model)

    # ... your training loop ...

    # At the end of each period (pass the accuracy improvement for this period):
    tracker.period_end(period_id, model, acc_delta=post_score - pre_score)

    # After all periods:
    tracker.save(out_dir / "memory_log.json")

Platform support
----------------
  CUDA  : torch.cuda.max_memory_allocated() / reset_peak_memory_stats()
  MPS   : torch.mps.current_allocated_memory()  (no peak API in PyTorch < 2.2;
          we poll before / after training and take the max)
  CPU   : reports 0 MB for all memory metrics (still tracks params + time)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_params(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _mem_now_mb(device: str) -> float:
    """Current allocated memory in MB on *device*."""
    if device.startswith("cuda"):
        return torch.cuda.memory_allocated() / 1024 ** 2
    if device == "mps" and hasattr(torch, "mps"):
        try:
            return torch.mps.current_allocated_memory() / 1024 ** 2
        except Exception:
            pass
    return 0.0


def _reset_peak(device: str) -> None:
    """Reset peak memory stats (CUDA only; MPS has no peak API)."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: str) -> float:
    """Peak memory since last reset in MB."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0    # MPS: caller tracked via polling


def _infer_mb(model: nn.Module, device: str) -> float:
    """Model static memory: difference in allocated memory before/after a dummy forward.

    Uses a minimal dummy input (batch=1, seq_len=4) so the measurement is of
    model weights + buffers only, not activation memory.
    """
    mem_before = _mem_now_mb(device)
    # Put model in eval mode, run a single dummy forward, measure delta
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy_ids  = torch.zeros((1, 4), dtype=torch.long, device=device)
            dummy_mask = torch.ones((1, 4), dtype=torch.long, device=device)
            if hasattr(model, "generate"):
                model.generate(
                    input_ids=dummy_ids,
                    attention_mask=dummy_mask,
                    max_new_tokens=2,
                )
            else:
                model(input_ids=dummy_ids, attention_mask=dummy_mask)
    except Exception:
        pass
    mem_after = _mem_now_mb(device)
    if was_training:
        model.train()
    return max(0.0, mem_after - mem_before)


# ── MemoryTracker ─────────────────────────────────────────────────────────────

class MemoryTracker:
    """Collects per-period memory + efficiency metrics for Paper B.

    Parameters
    ----------
    device : "cuda" | "mps" | "cpu"
    method : run identifier — "inca" | "b6_llama_pro" | "b1_finetune" | …
    """

    def __init__(self, device: str, method: str = "unknown") -> None:
        self.device   = device
        self.method   = method
        self._records: List[Dict[str, Any]] = []

        # State maintained between period_start and period_end
        self._cur_period:           Optional[str]   = None
        self._t_start:              Optional[float] = None
        self._mem_before_train:     float            = 0.0
        self._mem_peak_poll:        float            = 0.0
        self._params_before:        int              = 0

        # Cross-period state
        self._param_total_start:    int  = 0   # base model size (set at first period_start)
        self._cumulative_delta:     int  = 0   # running sum of all param_delta values
        self._first_period_seen:    bool = False

    # ── per-period API ────────────────────────────────────────────────────────

    def period_start(self, period_id: str, model: nn.Module) -> None:
        """Call immediately before the training loop for *period_id*."""
        self._cur_period = period_id
        total, trainable = _count_params(model)
        self._params_before    = trainable
        self._mem_before_train = _mem_now_mb(self.device)
        self._mem_peak_poll    = self._mem_before_train
        _reset_peak(self.device)
        self._t_start = time.perf_counter()
        # Capture base-model size once, at the very first period
        if not self._first_period_seen:
            self._param_total_start = total
            self._first_period_seen = True

    def poll(self) -> None:
        """Optional: call inside the training loop to update MPS peak estimate."""
        cur = _mem_now_mb(self.device)
        if cur > self._mem_peak_poll:
            self._mem_peak_poll = cur

    def period_end(
        self,
        period_id: str,
        model: nn.Module,
        acc_delta: float = 0.0,
    ) -> None:
        """Call immediately after the training loop for *period_id*.

        Parameters
        ----------
        period_id : must match the period_id passed to period_start()
        model     : the model at the end of the period (may have grown new blocks)
        acc_delta : accuracy improvement this period (post_acc - pre_acc)
        """
        wall_time_s   = time.perf_counter() - (self._t_start or time.perf_counter())
        total, trainable = _count_params(model)

        # Peak memory: CUDA uses hardware peak; MPS uses polled max
        if self.device.startswith("cuda"):
            peak_mb = _peak_mb(self.device)
        else:
            peak_mb = self._mem_peak_poll    # best estimate on MPS / CPU

        # Inference memory (static model footprint, no activations)
        inf_mb = _infer_mb(model, self.device)

        # Parameter delta vs period start
        param_delta = trainable - self._params_before
        self._cumulative_delta += param_delta

        # Growth percentage relative to base model
        param_growth_pct = (
            (total - self._param_total_start) / max(self._param_total_start, 1) * 100
        )

        record: Dict[str, Any] = {
            "method":                self.method,
            "period":                period_id,
            "peak_train_mb":         round(peak_mb, 2),
            "infer_mb":              round(inf_mb, 2),
            "param_total":           total,
            "param_total_start":     self._param_total_start,
            "param_trainable":       trainable,
            "param_delta":           param_delta,
            "cumulative_param_delta": self._cumulative_delta,
            "param_growth_pct":      round(param_growth_pct, 3),
            "wall_time_s":           round(wall_time_s, 2),
            "acc_delta":             round(acc_delta, 6),
            # derived: accuracy gain per MB of peak training memory
            "acc_per_mb":            round(acc_delta / max(peak_mb, 1e-6), 8),
        }
        self._records.append(record)
        self._cur_period = None

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: "str | Path") -> None:
        """Write all period records to *path* as a JSON array."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)
        print(f"[MemoryTracker] Saved {len(self._records)} period records → {path}")

    def summary(self) -> str:
        """Return a human-readable table of all recorded periods."""
        if not self._records:
            return "[MemoryTracker] No records yet."
        lines = [
            f"{'Period':<16} {'peak_MB':>9} {'infer_MB':>10} "
            f"{'params_train':>14} {'param_Δ':>9} {'cum_Δ':>10} {'growth%':>8} "
            f"{'time_s':>8} {'acc_Δ':>8} {'acc/MB':>10}"
        ]
        lines.append("-" * 110)
        for r in self._records:
            lines.append(
                f"{r['period']:<16} {r['peak_train_mb']:>9.1f} {r['infer_mb']:>10.1f} "
                f"{r['param_trainable']:>14,} {r['param_delta']:>+9,} "
                f"{r['cumulative_param_delta']:>+10,} {r['param_growth_pct']:>7.2f}% "
                f"{r['wall_time_s']:>8.1f} {r['acc_delta']:>8.4f} {r['acc_per_mb']:>10.6f}"
            )
        # Final row: cumulative totals
        if self._records:
            last = self._records[-1]
            total_acc = sum(r["acc_delta"] for r in self._records)
            lines.append("-" * 110)
            lines.append(
                f"{'TOTAL':<16} {'':>9} {'':>10} "
                f"{'base=' + str(self._param_total_start):>14} {'':>9} "
                f"{last['cumulative_param_delta']:>+10,} {last['param_growth_pct']:>7.2f}% "
                f"{'':>8} {total_acc:>8.4f}"
            )
        return "\n".join(lines)

    def param_comparison_table(self) -> str:
        """Print a LLaMA-Pro vs INCA parameter comparison table.

        Shows how INCA's adaptive parameter growth compares to LLaMA-Pro's fixed
        schedule, both in absolute terms and as a percentage of the base model.

        LLaMA-Pro reference numbers (Wu et al., 2024):
          Base:  LLaMA2-7B   ≈ 6 738M  (7B)
          Final: LLaMA-Pro   ≈ 8 303M  (8.3B)
          Added: ≈ 1 565M    (+23.2%), fixed schedule, all 8 blocks at once
        """
        if not self._records:
            return "[MemoryTracker] No records yet — call period_end() first."

        last = self._records[-1]
        base_m    = self._param_total_start / 1e6
        final_m   = last["param_total"] / 1e6
        added_m   = last["cumulative_param_delta"] / 1e6
        growth    = last["param_growth_pct"]

        # LLaMA-Pro reference
        LP_BASE_M  = 6_738.0   # LLaMA2-7B  [M]
        LP_FINAL_M = 8_303.0   # LLaMA-Pro  [M]
        LP_ADDED_M = LP_FINAL_M - LP_BASE_M
        LP_GROWTH  = LP_ADDED_M / LP_BASE_M * 100

        lines = [
            "",
            "┌─ Parameter Comparison: INCA vs LLaMA-Pro ──────────────────────────────────┐",
            f"│ {'Metric':<34} {'INCA (' + self.method + ')':>18}  {'LLaMA-Pro':>14} │",
            "├─────────────────────────────────────────────────────────────────────────────┤",
            f"│ {'Base model params (M)':<34} {base_m:>18.1f}  {LP_BASE_M:>14.1f} │",
            f"│ {'Final model params (M)':<34} {final_m:>18.1f}  {LP_FINAL_M:>14.1f} │",
            f"│ {'Params added (M)':<34} {added_m:>+18.1f}  {LP_ADDED_M:>+14.1f} │",
            f"│ {'Growth vs base (%)':<34} {growth:>17.2f}%  {LP_GROWTH:>13.2f}% │",
            f"│ {'Expansion schedule':<34} {'adaptive (saturation)':>18}  {'fixed (pre-train)':>14} │",
            f"│ {'Periods trained':<34} {len(self._records):>18}  {'1 (all at once)':>14} │",
            "└─────────────────────────────────────────────────────────────────────────────┘",
            "",
            "Per-period parameter growth (INCA):",
        ]

        for r in self._records:
            delta_m = r["param_delta"] / 1e6
            cum_m   = r["cumulative_param_delta"] / 1e6
            lines.append(
                f"  {r['period']:<16}  Δ={delta_m:>+7.1f}M  cumulative={cum_m:>+7.1f}M  "
                f"growth={r['param_growth_pct']:>6.2f}%  acc_Δ={r['acc_delta']:>+.4f}"
            )
        return "\n".join(lines)

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)
