"""Unified baseline runner for Phase 0 (B1 – B7).

Runs one or all of the seven baselines against a chosen dataset and writes
results to ``Phase0/results/<baseline_id>/``.

Usage
-----
Run a single baseline::

    python Phase0/scripts/run_baseline.py --method naive
    python Phase0/scripts/run_baseline.py --method ewc --lambda_ewc 100
    python Phase0/scripts/run_baseline.py --method lora_moe --lora_rank 8

Run all seven in sequence::

    python Phase0/scripts/run_baseline.py --method all

Override any config value on the CLI::

    python Phase0/scripts/run_baseline.py --method all \\
        --dataset cc_news \\
        --model_name distilgpt2 \\
        --max_periods 4 \\
        --epochs_per_period 3 \\
        --batch_size 16

Method names
------------
  naive      →  B1 — sequential fine-tuning (forgetting floor)
  replay     →  B2 — experience replay
  ewc        →  B3 — Elastic Weight Consolidation
  l2p        →  B4 — Learning to Prompt (frozen backbone + prompt pool)
  lora_moe   →  B5 — LoRA Mixture-of-Experts
  llama_pro  →  B6 — LLaMA-Pro-style vertical block expansion
  pnn        →  B7 — Progressive Neural Network (one column per period)
  all        →  run B1 → B7 in sequence

Results
-------
Each run writes to ``Phase0/results/<baseline_id>/``:
  metrics.json            per-period pre/post ppl, probe_acc, combined, RIR
  summary.json            ACC, BWT, FWT, final_combined_last_period
  training.log            timestamped console output
  config.snapshot.json    full config + baseline extras at run time
  probes_period<N>.csv    per-probe results for period N
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sure the repo root is on the path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Phase0.common.config import Phase0Config, load_config


# ── Method registry ──────────────────────────────────────────────────────────

METHOD_ALIASES = {
    "naive":     "b1_finetune",
    "replay":    "b2_replay",
    "ewc":       "b3_ewc",
    "l2p":       "b4_l2p",
    "lora_moe":  "b5_lora_moe",
    "llama_pro": "b6_llama_pro",
    "pnn":       "b7_pnn",
    # also accept the canonical names directly
    "b1_finetune":  "b1_finetune",
    "b2_replay":    "b2_replay",
    "b3_ewc":       "b3_ewc",
    "b4_l2p":       "b4_l2p",
    "b5_lora_moe":  "b5_lora_moe",
    "b6_llama_pro": "b6_llama_pro",
    "b7_pnn":       "b7_pnn",
}

ALL_METHODS = ["b1_finetune", "b2_replay", "b3_ewc", "b4_l2p",
               "b5_lora_moe", "b6_llama_pro", "b7_pnn"]

METHOD_LABELS = {
    "b1_finetune":  "B1  Naive fine-tuning",
    "b2_replay":    "B2  Experience replay",
    "b3_ewc":       "B3  Elastic Weight Consolidation (EWC)",
    "b4_l2p":       "B4  Learning to Prompt (L2P)",
    "b5_lora_moe":  "B5  LoRA Mixture-of-Experts",
    "b6_llama_pro": "B6  LLaMA-Pro block expansion",
    "b7_pnn":       "B7  Progressive Neural Network (PNN)",
}


# ── Per-method baseline constructors ─────────────────────────────────────────

def _build_baseline(method_id: str, cfg: Phase0Config,
                    method_args: Dict[str, Any]):
    """Import the right module and instantiate the baseline dataclass."""
    import importlib
    mod = importlib.import_module(f"Phase0.baselines.{method_id}")

    if method_id == "b1_finetune":
        return mod.B1NaiveFinetune(cfg=cfg)

    if method_id == "b2_replay":
        return mod.B2ReplayOnly(
            cfg=cfg,
            buffer_size=method_args.get("buffer_size", 2000),
            replay_ratio=method_args.get("replay_ratio", 0.5),
        )

    if method_id == "b3_ewc":
        return mod.B3EWC(
            cfg=cfg,
            lambda_ewc=method_args.get("lambda_ewc", 100.0),
        )

    if method_id == "b4_l2p":
        return mod.B4L2P(
            cfg=cfg,
            pool_size=method_args.get("pool_size", 10),
            prompt_len=method_args.get("prompt_len", 5),
            top_n=method_args.get("top_n", 3),
        )

    if method_id == "b5_lora_moe":
        return mod.B5LoRAMoE(
            cfg=cfg,
            lora_rank=method_args.get("lora_rank", 8),
            lora_alpha=method_args.get("lora_alpha", 16.0),
        )

    if method_id == "b6_llama_pro":
        return mod.B6LLaMAProExpansion(cfg=cfg)

    if method_id == "b7_pnn":
        return mod.B7PNN(cfg=cfg)

    raise ValueError(f"Unknown method_id: {method_id!r}")


# ── Single-method runner ──────────────────────────────────────────────────────

def run_one(method_id: str, cfg: Phase0Config,
            method_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from Phase0.common.runner import BaselineRunner

    label = METHOD_LABELS.get(method_id, method_id)
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  dataset={cfg.dataset}  model={cfg.model_name}  "
          f"periods={cfg.max_periods}  seed={cfg.seed}")
    print(f"{'='*72}")

    t0 = time.time()
    try:
        baseline = _build_baseline(method_id, cfg, method_args)
        runner = BaselineRunner(cfg, baseline)
        records = runner.run()
        elapsed = time.time() - t0
        print(f"\n  ✓ {method_id} completed in {elapsed:.0f}s")
        return records
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ✗ {method_id} FAILED after {elapsed:.0f}s: {exc}", file=sys.stderr)
        return None


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 0 — unified baseline runner (B1–B7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Method selection
    p.add_argument(
        "--method", required=True,
        metavar="METHOD",
        help="Baseline to run: naive | replay | ewc | l2p | lora_moe | "
             "llama_pro | pnn | all",
    )

    # Config + shared overrides
    p.add_argument("--config", default=None,
                   help="Path to YAML/JSON config (default: Phase0/configs/base.yaml)")
    p.add_argument("--dataset",            default=None)
    p.add_argument("--model_name",         default=None)
    p.add_argument("--max_periods",        type=int,   default=None)
    p.add_argument("--max_docs_per_period",type=int,   default=None)
    p.add_argument("--epochs_per_period",  type=int,   default=None)
    p.add_argument("--batch_size",         type=int,   default=None)
    p.add_argument("--lr",                 type=float, default=None)
    p.add_argument("--max_seq_len",        type=int,   default=None)
    p.add_argument("--seed",               type=int,   default=None)
    p.add_argument("--device",             default=None)

    # Method-specific hyperparameters
    p.add_argument("--lambda_ewc",   type=float, default=None, help="B3 EWC regularisation weight")
    p.add_argument("--buffer_size",  type=int,   default=None, help="B2 replay buffer size")
    p.add_argument("--replay_ratio", type=float, default=None, help="B2 replay fraction per minibatch")
    p.add_argument("--pool_size",    type=int,   default=None, help="B4 L2P prompt pool size")
    p.add_argument("--prompt_len",   type=int,   default=None, help="B4 L2P prompt token length")
    p.add_argument("--top_n",        type=int,   default=None, help="B4 L2P top-N keys to retrieve")
    p.add_argument("--lora_rank",    type=int,   default=None, help="B5 LoRA rank per expert")
    p.add_argument("--lora_alpha",   type=float, default=None, help="B5 LoRA alpha scaling")

    args = p.parse_args()

    # Resolve config path default
    config_path = args.config
    if config_path is None:
        default_cfg = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)

    # Build shared config with CLI overrides
    shared_overrides = {k: getattr(args, k)
                        for k in ("dataset", "model_name", "max_periods",
                                  "max_docs_per_period", "epochs_per_period",
                                  "batch_size", "lr", "max_seq_len", "seed", "device")
                        if getattr(args, k) is not None}
    cfg = load_config(config_path, overrides=shared_overrides)

    # Method-specific args (only non-None values passed through)
    method_args: Dict[str, Any] = {
        k: getattr(args, k)
        for k in ("lambda_ewc", "buffer_size", "replay_ratio", "pool_size",
                  "prompt_len", "top_n", "lora_rank", "lora_alpha")
        if getattr(args, k) is not None
    }

    # Resolve method(s)
    raw_method = args.method.lower().strip()
    if raw_method == "all":
        methods = ALL_METHODS
    else:
        if raw_method not in METHOD_ALIASES:
            print(f"Unknown method {raw_method!r}. Choose from: "
                  f"all, {', '.join(METHOD_ALIASES.keys())}", file=sys.stderr)
            sys.exit(1)
        methods = [METHOD_ALIASES[raw_method]]

    print(f"\nPhase 0 baseline runner")
    print(f"  methods  : {[METHOD_LABELS[m] for m in methods]}")
    print(f"  dataset  : {cfg.dataset}")
    print(f"  model    : {cfg.model_name}")
    print(f"  periods  : {cfg.max_periods}")
    print(f"  epochs   : {cfg.epochs_per_period}")
    print(f"  batch    : {cfg.batch_size}")
    print(f"  seed     : {cfg.seed}")
    if method_args:
        print(f"  method args: {method_args}")

    t_total = time.time()
    failures: List[str] = []

    for method_id in methods:
        result = run_one(method_id, cfg, method_args)
        if result is None:
            failures.append(method_id)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*72}")
    print(f"Total wall time: {elapsed_total:.0f}s")
    if failures:
        print(f"FAILED ({len(failures)}/{len(methods)}): {failures}", file=sys.stderr)
        sys.exit(1)
    else:
        ran = len(methods)
        print(f"All {ran} baseline(s) completed. Results in Phase0/results/")


if __name__ == "__main__":
    main()
