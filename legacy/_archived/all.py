"""Download all CAPSEL Phase-0 raw datasets from HuggingFace.

Runs each download_<dataset>.py script in sequence.  A dataset is skipped
automatically if its raw/ directory already contains at least one non-empty
.parquet or .jsonl file.  Pass --force to override this check.

Output: Phase0/data/raw/<dataset>/   (Parquet or JSONL files)

Next step after this completes:
    python Phase0/data/preprocess_all.py

Usage::

    python Phase0/data/download_all.py
    python Phase0/data/download_all.py --only realtimeqa temporalwiki
    python Phase0/data/download_all.py --skip tic_lm cc_news
    python Phase0/data/download_all.py --force
    python Phase0/data/download_all.py --max_docs_per_period 500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

HERE     = Path(__file__).resolve().parent
RAW_ROOT = HERE / "raw"

# Ordered list of all downloadable datasets.
# streaming_qa is listed here but its downloader delegates to download_cc_news.py.
ALL_DATASETS = [
    "realtimeqa",
    "temporalwiki",
    "streaming_qa",   # delegates to download_cc_news.py internally
    "cc_news",
    "trace",
    "ckl",
    "medmcqa",
    "tic_lm",
]

# Datasets skipped by --quick (large downloads)
HEAVY = {"cc_news", "trace", "tic_lm"}


def _has_raw_data(name: str) -> bool:
    """True if raw/<name>/ has at least one non-empty .parquet or .jsonl file."""
    raw_path = RAW_ROOT / name
    if not raw_path.is_dir():
        return False
    return any(
        f.is_file() and f.suffix in (".parquet", ".jsonl") and f.stat().st_size > 0
        for f in raw_path.rglob("*")
    )


def _raw_summary(name: str) -> str:
    raw_path = RAW_ROOT / name
    files = [
        f for f in raw_path.rglob("*")
        if f.is_file() and f.suffix in (".parquet", ".jsonl") and f.stat().st_size > 0
    ]
    total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
    return f"{len(files)} file(s), {total_mb:.1f} MB"


def _run_downloader(name: str, extra_args: List[str]) -> int:
    script = HERE / f"download_{name}.py"
    if not script.exists():
        print(f"  [skip] download_{name}.py not found.")
        return 0
    return subprocess.call([sys.executable, str(script)] + extra_args, cwd=str(HERE))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download all CAPSEL Phase-0 raw datasets from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", nargs="*", default=None, metavar="DS",
                   help="Subset of datasets to download (default: all).")
    p.add_argument("--skip", nargs="*", default=[], metavar="DS",
                   help="Datasets to skip.")
    p.add_argument("--quick", action="store_true",
                   help="Skip heavy datasets (cc_news, trace, tic_lm).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if raw data already exists.")
    p.add_argument("--max_docs_per_period", type=int, default=0, metavar="N",
                   help="Passed through to each downloader.")
    p.add_argument("--max_periods", type=int, default=0, metavar="N",
                   help="Passed through to each downloader.")
    args = p.parse_args()

    skip = set(args.skip)
    if args.quick:
        skip |= HEAVY

    targets = [d for d in (args.only or ALL_DATASETS) if d not in skip]

    # Build extra args to forward to each downloader
    extra: List[str] = []
    if args.max_docs_per_period > 0:
        extra += ["--max_docs_per_period", str(args.max_docs_per_period)]
    if args.max_periods > 0:
        extra += ["--max_periods", str(args.max_periods)]

    print(f"[download_all] targets : {targets}")
    if extra:
        print(f"[download_all] extra   : {' '.join(extra)}")

    skipped  = []
    failed   = []

    for name in targets:
        # streaming_qa raw lives under cc_news — check that directory instead
        check_name = "cc_news" if name == "streaming_qa" else name

        if not args.force and _has_raw_data(check_name):
            print(f"\n  · {name}: already downloaded "
                  f"({_raw_summary(check_name)}) — skipping  "
                  f"(--force to re-download)")
            skipped.append(name)
            continue

        print(f"\n{'='*60}\n  Downloading: {name}\n{'='*60}")
        rc = _run_downloader(name, extra)
        if rc != 0:
            failed.append((name, rc))
            print(f"  ✗ {name} failed (exit {rc})")
        else:
            print(f"  ✓ {name} done")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if skipped:
        print(f"[download_all] skipped (already present) : {skipped}")
    done = [t for t in targets if t not in skipped and t not in [f for f,_ in failed]]
    if done:
        print(f"[download_all] downloaded : {done}")
    if failed:
        print(f"[download_all] FAILED : {failed}")
        sys.exit(1)
    print(f"[download_all] all done.")
    print(f"[download_all] next step: python Phase0/data/preprocess_all.py")


if __name__ == "__main__":
    main()
