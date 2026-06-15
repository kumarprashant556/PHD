"""Full CAPSEL Phase-0 data pipeline: download then preprocess.

This is a thin orchestrator that calls download_all.py followed by
preprocess_all.py.  Run the two scripts individually if you want finer
control over which step to execute.

Pipeline
--------
  1. download_all.py   — fetch raw data from HuggingFace → raw/<dataset>/
  2. preprocess_all.py — convert raw files → processed/<dataset>/

Both steps skip datasets that are already complete (raw files present /
timeline.json exists).  Pass --force to override.

Usage::

    # Full pipeline (skips already-downloaded / already-preprocessed datasets)
    python Phase0/data/prepare_all.py

    # Download only
    python Phase0/data/prepare_all.py --download-only

    # Preprocess only (raw files already present)
    python Phase0/data/prepare_all.py --preprocess-only

    # Only specific datasets, end-to-end
    python Phase0/data/prepare_all.py --only realtimeqa temporalwiki

    # Force re-run everything
    python Phase0/data/prepare_all.py --force

    # Limit size (useful for smoke tests)
    python Phase0/data/prepare_all.py \\
        --max_periods 4 --max_docs_per_period 200 --max_probes_per_period 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent


def _call(script: str, extra: List[str]) -> int:
    return subprocess.call(
        [sys.executable, str(HERE / script)] + extra,
        cwd=str(HERE),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Full CAPSEL Phase-0 data pipeline: download then preprocess.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--download-only",   dest="download_only",
                   action="store_true",
                   help="Run download_all.py only; skip preprocessing.")
    p.add_argument("--preprocess-only", dest="preprocess_only",
                   action="store_true",
                   help="Run preprocess_all.py only; skip downloading.")
    p.add_argument("--only", nargs="*", default=None, metavar="DS",
                   help="Subset of datasets to process end-to-end.")
    p.add_argument("--skip", nargs="*", default=[], metavar="DS",
                   help="Datasets to exclude.")
    p.add_argument("--quick", action="store_true",
                   help="Skip heavy datasets (cc_news, trace, tic_lm) and "
                        "apply small caps.")
    p.add_argument("--force", action="store_true",
                   help="Re-download and re-preprocess even if outputs exist.")
    p.add_argument("--max_periods",           type=int, default=0, metavar="N")
    p.add_argument("--max_docs_per_period",   type=int, default=0, metavar="N")
    p.add_argument("--max_probes_per_period", type=int, default=0, metavar="N")
    args = p.parse_args()

    if args.quick:
        if args.max_periods == 0:
            args.max_periods = 4
        if args.max_docs_per_period == 0:
            args.max_docs_per_period = 200
        if args.max_probes_per_period == 0:
            args.max_probes_per_period = 200
        if args.skip is None:
            args.skip = []
        args.skip = list(set(args.skip) | {"cc_news", "trace", "tic_lm"})

    # ── Build forwarded args ──────────────────────────────────────────────────
    fwd: List[str] = []
    if args.only:
        fwd += ["--only"] + args.only
    if args.skip:
        fwd += ["--skip"] + args.skip
    if args.force:
        fwd += ["--force"]
    if args.max_periods > 0:
        fwd += ["--max_periods", str(args.max_periods)]
    if args.max_docs_per_period > 0:
        fwd += ["--max_docs_per_period", str(args.max_docs_per_period)]
    if args.max_probes_per_period > 0:
        fwd += ["--max_probes_per_period", str(args.max_probes_per_period)]

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not args.preprocess_only:
        print("=" * 60)
        print("  STEP 1 / 2 — download_all.py")
        print("=" * 60)
        rc = _call("download_all.py", fwd)
        if rc != 0:
            print(f"[prepare_all] download_all.py failed (exit {rc}). Aborting.")
            sys.exit(rc)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    if not args.download_only:
        print("\n" + "=" * 60)
        print("  STEP 2 / 2 — preprocess_all.py")
        print("=" * 60)
        rc = _call("preprocess_all.py", fwd)
        if rc != 0:
            print(f"[prepare_all] preprocess_all.py failed (exit {rc}).")
            sys.exit(rc)

    print("\n[prepare_all] pipeline complete.")


if __name__ == "__main__":
    main()
