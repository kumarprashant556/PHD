"""Run B1-B7 baselines sequentially under one shared timestamped directory.

Usage
-----
    # all 7 baselines, default config:
    python scripts/train_baselines.py --config configs/base.yaml --device mps

    # subset:
    python scripts/train_baselines.py --config configs/base.yaml \\
        --baselines b1,b2,b3 --device mps

    # custom output root:
    python scripts/train_baselines.py --config configs/base.yaml \\
        --out-root results/sweep_$(date +%Y%m%d_%H%M%S)

Each baseline gets its own subdirectory under ``--out-root`` (default
``results/sweep_<timestamp>/``), with the standard artifacts:

    <sweep>/
      sweep.log                       — combined log of the whole sweep
      <baseline>_<ts>/
          run.log
          config.json
          loss_curve_<pid>.json
          regret_matrix.csv
          metrics_summary.json
          <baseline>_best/            — HF-format best checkpoint

A final ``sweep_summary.json`` aggregates BWT / ACC / FWT across baselines.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

# Repo root on sys.path so `from baselines._runtime ...` resolves.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines._runtime import INCAConfig, TrainerRunner       # noqa: E402

# Baselines registry: id → (module, baseline class)
from baselines.b1_finetune  import B1NaiveFinetune             # noqa: E402
from baselines.b2_replay    import B2Replay                    # noqa: E402
from baselines.b3_ewc       import B3EWC                       # noqa: E402
from baselines.b4_l2p       import B4L2P                       # noqa: E402
from baselines.b5_lora_moe  import B5LoRAMoE                   # noqa: E402
from baselines.b6_llama_pro import B6LLaMAProExpansion         # noqa: E402
from baselines.b7_pnn       import B7PNN                       # noqa: E402


BASELINES = {
    "b1": ("b1_finetune",  B1NaiveFinetune),
    "b2": ("b2_replay",    B2Replay),
    "b3": ("b3_ewc",       B3EWC),
    "b4": ("b4_l2p",       B4L2P),
    "b5": ("b5_lora_moe",  B5LoRAMoE),
    "b6": ("b6_llama_pro", B6LLaMAProExpansion),
    "b7": ("b7_pnn",       B7PNN),
}


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _make_cfg(cfg_dict: dict, out_root: Path) -> INCAConfig:
    """Filter cfg dict to INCAConfig fields and inject ``out_dir``."""
    cfg_dict = dict(cfg_dict)
    cfg_dict["out_dir"] = str(out_root)
    return INCAConfig(**{
        k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__
    })


def _setup_sweep_logger(out_root: Path) -> logging.Logger:
    out_root.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("capsel.sweep")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(out_root / "sweep.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    return log


def main() -> None:
    p = argparse.ArgumentParser(description="Run B1-B7 CL baselines sequentially")
    p.add_argument("--config",   required=True, help="YAML config (configs/base.yaml)")
    p.add_argument("--baselines", default="b1,b2,b3,b4,b5,b6,b7",
                   help="Comma-separated baseline ids to run (default: all 7)")
    p.add_argument("--device",   default=None,  help="cuda | mps | cpu (default: auto)")
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--out-root", default=None,
                   help="Output root (default: results/sweep_<timestamp>/)")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep running remaining baselines if one fails")
    args = p.parse_args()

    cfg_dict = _load_config(args.config)
    if args.seed is not None:
        cfg_dict["seed"] = args.seed

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else _ROOT / "results" / f"sweep_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    log = _setup_sweep_logger(out_root)
    log.info("=" * 64)
    log.info("Sweep started: out_root=%s", out_root)
    log.info("Config: %s", args.config)
    log.info("Baselines: %s", args.baselines)
    log.info("Device: %s", args.device or "(auto)")
    log.info("=" * 64)

    requested = [b.strip().lower() for b in args.baselines.split(",") if b.strip()]
    unknown = [b for b in requested if b not in BASELINES]
    if unknown:
        raise SystemExit(f"Unknown baseline id(s): {unknown}.  Choose from {list(BASELINES)}")

    results: List[Dict[str, object]] = []
    sweep_t0 = time.time()

    for b_id in requested:
        b_name, b_class = BASELINES[b_id]
        log.info("")
        log.info("█" * 64)
        log.info("█  %s  (%s)", b_id.upper(), b_name)
        log.info("█" * 64)

        cfg = _make_cfg(cfg_dict, out_root)
        baseline = b_class(cfg=cfg)
        t0 = time.time()
        ok, err = True, None
        try:
            TrainerRunner(cfg, baseline, device=args.device).run()
        except Exception as e:                                  # noqa: BLE001
            ok, err = False, repr(e)
            log.exception("Baseline %s FAILED: %s", b_id, e)
            if not args.continue_on_error:
                raise

        results.append({
            "baseline_id":  b_id,
            "baseline_name": b_name,
            "ok":            ok,
            "error":         err,
            "duration_sec":  round(time.time() - t0, 1),
        })

    # ── Aggregate summary ──────────────────────────────────────────────────
    summary = {
        "started_at":    ts,
        "out_root":      str(out_root),
        "config":        args.config,
        "device":        args.device or "(auto)",
        "config_dump":   {k: v for k, v in cfg_dict.items() if not k.startswith("_")},
        "duration_sec":  round(time.time() - sweep_t0, 1),
        "baselines":     results,
    }
    # Attach each baseline's metrics_summary.json if it exists.
    for r in results:
        baseline_dirs = sorted(out_root.glob(f"{r['baseline_name']}_*"))
        if baseline_dirs:
            ms = baseline_dirs[-1] / "metrics_summary.json"
            if ms.exists():
                r["metrics"] = json.loads(ms.read_text())
                r["run_dir"] = str(baseline_dirs[-1].relative_to(out_root))

    with open(out_root / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("")
    log.info("=" * 64)
    log.info("SWEEP DONE — %.1f min total", summary["duration_sec"] / 60.0)
    log.info("=" * 64)
    for r in results:
        m = r.get("metrics", {})
        if r["ok"] and m:
            log.info("  %-14s  BWT=%+.4f  ACC=%.4f  FWT=%+.4f  (%.1f min)",
                     r["baseline_id"], m.get("BWT", 0), m.get("ACC", 0), m.get("FWT", 0),
                     r["duration_sec"] / 60.0)
        else:
            log.info("  %-14s  FAILED: %s", r["baseline_id"], r.get("error"))
    log.info("Sweep summary: %s", out_root / "sweep_summary.json")


if __name__ == "__main__":
    main()
