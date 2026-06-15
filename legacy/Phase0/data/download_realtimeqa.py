"""Download RealtimeQA raw dataset from HuggingFace.

Fetches the dataset and saves the raw Parquet file ONLY.
No preprocessing is performed here — run preprocess_realtimeqa.py next.

HF source : ``prajaktakini/realtime_qa``
Raw output : Phase0/data/raw/realtimeqa/train.parquet

Run::

    python Phase0/data/download_realtimeqa.py
    python Phase0/data/download_realtimeqa.py --hf_id prajaktakini/realtime_qa
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

from _utils import (
    raw_dir,
    require_hf_datasets,
    save_raw_parquet,
)

DATASET_NAME  = "realtimeqa"
DEFAULT_HF_ID = "prajaktakini/realtime_qa"


def main() -> None:
    p = argparse.ArgumentParser(description="Download RealtimeQA raw Parquet.")
    p.add_argument("--hf_id", default=DEFAULT_HF_ID,
                   help="HuggingFace dataset id.")
    args = p.parse_args()

    hf  = require_hf_datasets()
    raw = raw_dir(DATASET_NAME)

    print(f"[download:realtimeqa] loading {args.hf_id!r} …")
    ds = hf.load_dataset(args.hf_id)

    print(f"[download:realtimeqa] saving raw Parquet → {raw}")
    save_raw_parquet(ds, raw)

    files = list(raw.glob("*.parquet"))
    print(f"[download:realtimeqa] done — {len(files)} file(s) in {raw}")
    print(f"[download:realtimeqa] next step: python Phase0/data/preprocess_realtimeqa.py")


if __name__ == "__main__":
    main()
