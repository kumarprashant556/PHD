"""Download MedMCQA raw splits from HuggingFace.

Downloads all three splits (train, validation, test) from
``openlifescienceai/medmcqa`` and saves them as Parquet files under
``Phase0/data/raw/medmcqa/``.  No preprocessing is performed here.

HuggingFace:  ``openlifescienceai/medmcqa``
Splits saved:
  raw/medmcqa/train.parquet      (~76 MB, 182k rows, labelled)
  raw/medmcqa/validation.parquet (~1.4 MB, 4.2k rows, labelled)
  raw/medmcqa/test.parquet       (~0.8 MB, 6.1k rows, labels absent)

Next step::

    python Phase0/data/preprocess_medmcqa.py

Usage::

    python Phase0/data/download_medmcqa.py
    python Phase0/data/download_medmcqa.py --force
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

from _utils import raw_dir, require_hf_datasets, save_raw_parquet

DATASET_NAME = "medmcqa"
HF_ID        = "openlifescienceai/medmcqa"


def _has_raw(raw: Path) -> bool:
    """True if all three expected Parquet files are present and non-empty."""
    return all(
        (raw / f"{s}.parquet").is_file() and (raw / f"{s}.parquet").stat().st_size > 0
        for s in ("train", "validation", "test")
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download openlifescienceai/medmcqa raw Parquet splits."
    )
    p.add_argument("--force", action="store_true",
                   help="Re-download even if raw files already exist.")
    args = p.parse_args()

    raw = raw_dir(DATASET_NAME)

    if not args.force and _has_raw(raw):
        print(f"[download:medmcqa] already downloaded — skipping  (--force to re-download)")
        for s in ("train", "validation", "test"):
            mb = (raw / f"{s}.parquet").stat().st_size / 1024 / 1024
            print(f"  {s}.parquet  {mb:.1f} MB")
        print("[download:medmcqa] next step: python Phase0/data/preprocess_medmcqa.py")
        return

    hf = require_hf_datasets()
    print(f"[download:medmcqa] loading {HF_ID!r} from HuggingFace…")
    ds = hf.load_dataset(HF_ID)

    print(f"[download:medmcqa] saving raw Parquet splits to {raw} …")
    save_raw_parquet(ds, raw)

    print(f"[download:medmcqa] done.")
    print(f"[download:medmcqa] next step: python Phase0/data/preprocess_medmcqa.py")


if __name__ == "__main__":
    main()
