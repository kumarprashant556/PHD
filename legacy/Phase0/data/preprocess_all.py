"""Preprocess all CAPSEL Phase-0 raw datasets into period JSONL.

Runs each preprocess_<dataset>.py script in sequence.  Each preprocessor
reads from Phase0/data/raw/<dataset>/ and writes to
Phase0/data/processed/<dataset>/.

A dataset is skipped automatically if its processed/ directory already
contains a timeline.json (meaning it has been preprocessed before).
Pass --force to override and re-run preprocessing.

Usage::

    python Phase0/data/preprocess_all.py
    python Phase0/data/preprocess_all.py --only realtimeqa temporalwiki
    python Phase0/data/preprocess_all.py --force
    python Phase0/data/preprocess_all.py --max_docs_per_period 200 \\
        --max_probes_per_period 200 --max_periods 24
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

HERE      = Path(__file__).resolve().parent
PROC_ROOT = HERE / "processed"

# Datasets that have a dedicated preprocess_<dataset>.py script.
ALL_DATASETS = [
    "realtimeqa",
    "temporalwiki",
    "streaming_qa",
    "medmcqa",
]


def _is_preprocessed(name: str) -> bool:
    """True if processed/<name>/timeline.json exists (preprocessing done)."""
    return (PROC_ROOT / name / "timeline.json").exists()


def _proc_summary(name: str) -> str:
    proc = PROC_ROOT / name
    n_probe_files = len(list((proc / "probes").glob("*.jsonl"))) if (proc / "probes").exists() else 0
    n_stream_files = len(list((proc / "stream").glob("*.jsonl"))) if (proc / "stream").exists() else 0
    return f"{n_probe_files} probe file(s), {n_stream_files} stream file(s)"


def _run_preprocessor(name: str, extra_args: List[str]) -> int:
    script = HERE / f"preprocess_{name}.py"
    if not script.exists():
        print(f"  [skip] preprocess_{name}.py not found.")
        return 0
    return subprocess.call([sys.executable, str(script)] + extra_args, cwd=str(HERE))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess all CAPSEL Phase-0 datasets from raw files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", nargs="*", default=None, metavar="DS",
                   help="Subset of datasets to preprocess (default: all).")
    p.add_argument("--skip", nargs="*", default=[], metavar="DS",
                   help="Datasets to skip.")
    p.add_argument("--force", action="store_true",
                   help="Re-preprocess even if processed output already exists.")
    p.add_argument("--max_docs_per_period",   type=int, default=0, metavar="N",
                   help="Max stream docs per period (0 = no cap).")
    p.add_argument("--max_probes_per_period", type=int, default=0, metavar="N",
                   help="Max probes per period (0 = no cap).")
    p.add_argument("--max_periods",           type=int, default=0, metavar="N",
                   help="Max periods (0 = no cap; streaming_qa hard-capped at 48).")
    args = p.parse_args()

    skip    = set(args.skip)
    targets = [d for d in (args.only or ALL_DATASETS) if d not in skip]

    # Forward caps to each preprocessor
    extra: List[str] = []
    if args.max_docs_per_period > 0:
        extra += ["--max_docs_per_period", str(args.max_docs_per_period)]
    if args.max_probes_per_period > 0:
        extra += ["--max_probes_per_period", str(args.max_probes_per_period)]
    if args.max_periods > 0:
        extra += ["--max_periods", str(args.max_periods)]

    print(f"[preprocess_all] targets : {targets}")
    if extra:
        print(f"[preprocess_all] extra   : {' '.join(extra)}")

    skipped = []
    failed  = []

    for name in targets:
        if not args.force and _is_preprocessed(name):
            print(f"\n  · {name}: already preprocessed "
                  f"({_proc_summary(name)}) — skipping  "
                  f"(--force to re-run)")
            skipped.append(name)
            continue

        print(f"\n{'='*60}\n  Preprocessing: {name}\n{'='*60}")
        rc = _run_preprocessor(name, extra)
        if rc != 0:
            failed.append((name, rc))
            print(f"  ✗ {name} failed (exit {rc})")
        else:
            print(f"  ✓ {name} done")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if skipped:
        print(f"[preprocess_all] skipped (already done) : {skipped}")
    done = [t for t in targets if t not in skipped and t not in [f for f,_ in failed]]
    if done:
        print(f"[preprocess_all] preprocessed : {done}")
    if failed:
        print(f"[preprocess_all] FAILED : {failed}")
        sys.exit(1)
    print(f"[preprocess_all] all done.")


if __name__ == "__main__":
    main()
