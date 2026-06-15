"""Process CC-News and TiC-LM raw files into CAPSEL temporal data v2.

Examples
--------
Small CC-News smoke run:

    python preprocessing/run.py cc_news --force --max-docs-per-period 200

Full CC-News v2 processing from the local raw file:

    python preprocessing/run.py cc_news --force

TiC-LM daily slices:

    python preprocessing/run.py tic_lm --force

Write v2 output and also refresh the legacy loader directories:

    python preprocessing/run.py cc_news --force --write-legacy-copy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
_PREPROCESSING = Path(__file__).resolve().parent
if str(_PREPROCESSING) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING))

from temporal import (
    DEFAULT_CC_NEWS_RAW,
    DEFAULT_TIC_LM_RAW_DIR,
    REPO_ROOT,
    process_cc_news_raw,
    process_tic_lm_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create v2 CAPSEL temporal stream/probe files from raw JSONL data."
    )
    parser.add_argument(
        "dataset",
        choices=["cc_news", "tic_lm", "all"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--cc-news-raw",
        type=Path,
        default=DEFAULT_CC_NEWS_RAW,
        help=f"Raw CC-News JSONL file. Default: {DEFAULT_CC_NEWS_RAW}",
    )
    parser.add_argument(
        "--tic-lm-raw-dir",
        type=Path,
        default=DEFAULT_TIC_LM_RAW_DIR,
        help=f"Raw TiC-LM JSONL directory. Default: {DEFAULT_TIC_LM_RAW_DIR}",
    )
    parser.add_argument(
        "--cc-output-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "cc_news" / "processed",
        help="Where CC-News processed files are written.",
    )
    parser.add_argument(
        "--tic-output-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "tic_lm" / "processed",
        help="Where TiC-LM processed files are written.",
    )
    parser.add_argument(
        "--cc-period-granularity",
        choices=["month", "quarter", "half_year", "year"],
        default="half_year",
        help="CC-News period slicing. Default: half_year.",
    )
    parser.add_argument(
        "--tic-period-granularity",
        choices=["day", "month", "quarter", "half_year", "year"],
        default="day",
        help="TiC-LM period slicing. Default: day.",
    )
    parser.add_argument(
        "--max-docs-per-period",
        type=int,
        default=0,
        help="Uniform reservoir cap per period. 0 means all accepted documents.",
    )
    parser.add_argument(
        "--probes-per-period",
        type=int,
        default=300,
        help="Maximum generated probes per period.",
    )
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--min-sentences", type=int, default=3)
    parser.add_argument("--max-periods", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing v2 output.")
    parser.add_argument(
        "--write-legacy-copy",
        action="store_true",
        help="Also overwrite processed/stream, processed/probes, timeline.json, metadata.json.",
    )
    args = parser.parse_args()

    summaries: Dict[str, Any] = {}
    if args.dataset in {"cc_news", "all"}:
        summaries["cc_news"] = process_cc_news_raw(
            raw_path=args.cc_news_raw,
            output_root=args.cc_output_root,
            period_granularity=args.cc_period_granularity,
            max_docs_per_period=args.max_docs_per_period,
            probes_per_period=args.probes_per_period,
            min_words=args.min_words,
            min_chars=args.min_chars,
            min_sentences=args.min_sentences,
            max_periods=args.max_periods,
            seed=args.seed,
            force=args.force,
            write_legacy_copy=args.write_legacy_copy,
        )

    if args.dataset in {"tic_lm", "all"}:
        summaries["tic_lm"] = process_tic_lm_raw(
            raw_dir=args.tic_lm_raw_dir,
            output_root=args.tic_output_root,
            period_granularity=args.tic_period_granularity,
            max_docs_per_period=args.max_docs_per_period,
            probes_per_period=args.probes_per_period,
            min_words=args.min_words,
            min_chars=args.min_chars,
            min_sentences=args.min_sentences,
            max_periods=args.max_periods,
            seed=args.seed,
            force=args.force,
            write_legacy_copy=args.write_legacy_copy,
        )

    for dataset, meta in summaries.items():
        n_examples = sum(period["stream_examples"] for period in meta["counts"].values())
        n_src_docs = sum(period["stream_source_docs"] for period in meta["counts"].values())
        n_probes = sum(period["probes"] for period in meta["counts"].values())
        print(
            f"{dataset}: {meta['n_periods']} periods, "
            f"{n_src_docs:,} source docs → {n_examples:,} training examples, "
            f"{n_probes:,} eval probes"
        )
        print(f"  stream: {meta['stream_dir']}")
        print(f"  probes: {meta['probes_dir']}")

    print(json.dumps({k: v["timeline"] for k, v in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
