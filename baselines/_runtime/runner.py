"""TrainerRunner — the per-period CL loop.

Each baseline implements the ``Baseline`` protocol below:

    build_model(tokenizer, device)             — load model, set requires_grad
    make_trainer(args, raw_items, tokenizer,   — return a Seq2SeqTrainer
                 period_label, period_idx)
    on_period_start(period_label, period_idx)  — optional hook (topology change)
    on_period_end(period_label, period_idx,    — optional hook (buffer update,
                  raw_items)                     EWC Fisher, etc.)
    scoring_model() -> nn.Module               — model used for cloze eval

For each period the runner does:
    FWT pre-eval  →  on_period_start  →  trainer.train()  →  on_period_end
        →  BWT row eval  →  save best checkpoint (if improved)

Artifacts written to ``results/<baseline>_<ts>/``:
    config.json                  config snapshot
    run.log                      stdout + transformers warnings
    loss_curve_<pid>.json        per-period step-by-step loss (from TrainerLogCallback)
    regret_matrix.csv            BWT/FWT matrix
    metrics_summary.json         BWT, ACC, FWT, best period / score / checkpoint
    <baseline>_best/             HF-format checkpoint of the best period
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import random
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from evaluation.metrics import (
    StreamAccuracyMatrix,
    bwt as _bwt, acc as _acc, fwt as _fwt,
)

from .data            import load_cc_news_v2, load_probes
from .eval            import eval_cloze_accuracy, pretty_matrix
from .logging_setup   import LOGGER_NAME, setup_logging, TrainerLogCallback
from .precision       import autocast_dtype
from .trainer_factory import build_training_args


# ── Baseline protocol ─────────────────────────────────────────────────────────

class Baseline(Protocol):
    """Interface every B1-B7 baseline implements."""
    name: str
    def build_model(self, tokenizer, device: str) -> Any: ...
    def make_trainer(
        self,
        args: Seq2SeqTrainingArguments,
        raw_items: List[Dict[str, Any]],
        tokenizer,
        period_label: str,
        period_idx: int,
    ) -> Seq2SeqTrainer: ...
    def on_period_start(self, period_label: str, period_idx: int) -> None: ...
    def on_period_end(
        self, period_label: str, period_idx: int,
        raw_items: List[Dict[str, Any]],
    ) -> None: ...
    def scoring_model(self) -> Any: ...


# ── Runner ────────────────────────────────────────────────────────────────────

class TrainerRunner:
    """Drives a Baseline through the CC-News v2 temporal stream."""

    def __init__(self, cfg, baseline: Baseline, device: Optional[str] = None):
        self.cfg = cfg
        self.baseline = baseline
        self.device = device or _auto_device()

    def run(self) -> None:
        cfg, baseline, device = self.cfg, self.baseline, self.device

        n_per_period = getattr(cfg, "n_per_period", 20_000)
        max_periods  = getattr(cfg, "max_periods",  None)
        seed         = getattr(cfg, "seed", 42)

        random.seed(seed)
        torch.manual_seed(seed)

        ts      = _dt.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(getattr(cfg, "out_dir", "results")) / f"{baseline.name}_{ts}"

        # ── Logging ───────────────────────────────────────────────────
        setup_logging(out_dir, level=logging.INFO, quiet_hf=True)
        log  = logging.getLogger(f"{LOGGER_NAME}.runner")
        sect = logging.getLogger(f"{LOGGER_NAME}.section")

        sect.info("=" * 64)
        sect.info("Baseline: %s | device=%s | precision=%s | optim=%s",
                  baseline.name, device,
                  getattr(cfg, "precision", "bf16"),
                  "adafactor" if getattr(cfg, "use_adafactor", False) else "adamw")
        sect.info("Output dir: %s", out_dir)
        sect.info("=" * 64)

        # Persist full config snapshot once.
        (out_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, default=str),
            encoding="utf-8",
        )

        # ── Load data ─────────────────────────────────────────────────
        log.info("Loading CC-News v2 stream (n_per_period=%d, max_periods=%s)",
                 n_per_period, max_periods)
        raw = load_cc_news_v2(n_per_period=n_per_period, seed=seed, max_periods=max_periods)
        period_ids = list(raw.keys())
        log.info("Loaded %d period(s): %s", len(period_ids), period_ids)

        # ── Tokenizer + model ─────────────────────────────────────────
        log.info("Loading tokenizer + model: %s", cfg.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        baseline.build_model(tokenizer, device)

        # ── Probes cache ──────────────────────────────────────────────
        probes_cache = {pid: load_probes(pid) for pid in period_ids}
        for pid in period_ids:
            n_cloze = sum(
                1 for p in probes_cache[pid]
                if p.get("probe_type") in ("entity_cloze", "date_cloze")
            )
            log.info("Probes %s: %d total, %d cloze (used for scoring)",
                     pid, len(probes_cache[pid]), n_cloze)

        # ── Regret matrix + best-ckpt tracking ────────────────────────
        mat = StreamAccuracyMatrix(matrix=[], labels=period_ids)
        # best_score = row-mean accuracy over all probes seen so far;
        # we keep ONE HF-format checkpoint (the highest-scoring period).
        best_score: float = -1.0
        best_pid:   str   = ""

        for idx, pid in enumerate(period_ids):
            items = raw[pid]
            sect.info("")
            sect.info("─── Period %d/%d: %s  (%d items) ───",
                      idx + 1, len(period_ids), pid, len(items))

            # FWT: eval THIS period's probes BEFORE training on it
            if idx > 0:
                fwt = eval_cloze_accuracy(
                    baseline.scoring_model(), probes_cache[pid],
                    tokenizer, device, max_input_len=cfg.max_input_length,
                    autocast_dtype=autocast_dtype(cfg, device),
                )
                mat.set(idx - 1, idx, fwt)
                log.info("FWT R[%d,%d] = %.4f  (model-after-%s on %s probes)",
                         idx - 1, idx, fwt, period_ids[idx - 1], pid)

            # Period-start hook (B5/B6/B7 grow model here)
            baseline.on_period_start(pid, idx)

            # Build Trainer + attach our log-capture callback.  Remove Trainer's
            # default PrinterCallback so it doesn't double-print raw log dicts.
            args = build_training_args(cfg, out_dir, pid, seed, device)
            trainer = baseline.make_trainer(args, items, tokenizer, pid, idx)
            try:
                from transformers.trainer_callback import PrinterCallback
                trainer.remove_callback(PrinterCallback)
            except Exception:
                pass
            cb = TrainerLogCallback(period_label=pid, period_idx=idx)
            trainer.add_callback(cb)

            log.info("Training period %s …", pid)
            t0 = _dt.now()
            result = trainer.train()
            dur = (_dt.now() - t0).total_seconds()
            log.info("Period %s done | train_loss=%.4f | %.1f s (%.1f items/s)",
                     pid, result.training_loss, dur,
                     len(items) * cfg.epochs_per_period / max(dur, 1e-6))

            # Persist per-period loss curve from the callback's history.
            (out_dir / f"loss_curve_{pid}.json").write_text(
                json.dumps(cb.history, indent=2, default=str),
                encoding="utf-8",
            )

            # Period-end hook (B2 buffer, B3 Fisher).
            baseline.on_period_end(pid, idx, items)

            # Flush device cache before the eval pass.
            if device == "mps":
                torch.mps.empty_cache()
            elif "cuda" in device:
                torch.cuda.empty_cache()

            # BWT row: eval on all periods 0..idx
            score_model = baseline.scoring_model()
            log.info("Evaluating model-after-%s on probes from all past periods …", pid)
            row_accs: List[float] = []
            for j, jpid in enumerate(period_ids[: idx + 1]):
                if jpid not in probes_cache:
                    continue
                acc = eval_cloze_accuracy(
                    score_model, probes_cache[jpid],
                    tokenizer, device, max_input_len=cfg.max_input_length,
                    autocast_dtype=autocast_dtype(cfg, device),
                )
                mat.set(idx, j, acc)
                row_accs.append(acc)
                tag = "diag" if j == idx else "BWT "
                log.info("  [%s] R[%d,%d] (%s → %s) = %.4f",
                         tag, idx, j, pid, jpid, acc)

            # Save only the best checkpoint across periods (HF-native format).
            period_score = sum(row_accs) / len(row_accs) if row_accs else 0.0
            if period_score > best_score:
                best_score = period_score
                best_pid   = pid
                best_dir   = out_dir / f"{baseline.name}_best"
                trainer.save_model(str(best_dir))
                log.info("New best checkpoint (score=%.4f, after %s) → %s/",
                         period_score, pid, best_dir.name)
            else:
                log.info("Score %.4f did not improve over best=%.4f (after %s); "
                         "keeping previous best checkpoint",
                         period_score, best_score, best_pid)

        # ── Final metrics ─────────────────────────────────────────────
        bwt_s = _bwt(mat); acc_s = _acc(mat); fwt_s = _fwt(mat)
        sect.info("")
        sect.info("=" * 64)
        sect.info("FINAL  %s   BWT=%+.4f   ACC=%.4f   FWT=%+.4f",
                  baseline.name, bwt_s, acc_s, fwt_s)
        sect.info("=" * 64)
        log.info("\n%s", pretty_matrix(mat, period_ids))

        # ── Write artifacts ───────────────────────────────────────────
        mat_path = out_dir / "regret_matrix.csv"
        with open(mat_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["after_period"] + [f"probes_{p}" for p in period_ids])
            for t_idx, t_pid in enumerate(period_ids):
                w.writerow(
                    [t_pid] + [f"{mat.get(t_idx, j):.4f}" for j in range(len(period_ids))]
                )

        summary = {
            "baseline": baseline.name, "n_periods": len(period_ids),
            "period_ids": period_ids,
            "BWT": round(bwt_s, 4), "ACC": round(acc_s, 4), "FWT": round(fwt_s, 4),
            "best_period": best_pid,
            "best_score": round(best_score, 4),
            "best_checkpoint": f"{baseline.name}_best/",
            "regret_matrix": mat.matrix,
        }
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        log.info("Done → %s", out_dir)


def _auto_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
