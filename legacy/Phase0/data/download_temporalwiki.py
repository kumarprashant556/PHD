"""Download TemporalWiki raw articles from HuggingFace (streaming).

Fetches two Wikipedia snapshots and saves raw JSONL files ONLY.
No preprocessing is performed here — run preprocess_temporalwiki.py next.

HF sources:
  period_2022 → ``seonghyeonye/TemporalWiki``   (2022-era Wikipedia text)
  period_2023 → ``wikimedia/wikipedia``           (November 2023 snapshot)

Raw output:
  Phase0/data/raw/temporalwiki/period_2022.jsonl
  Phase0/data/raw/temporalwiki/period_2023.jsonl

Each line: {"period": str, "title": str, "text": str}

Run::

    python Phase0/data/download_temporalwiki.py
    python Phase0/data/download_temporalwiki.py --max_docs_per_period 500
    python Phase0/data/download_temporalwiki.py --max_periods 1
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

from _utils import (
    cap,
    raw_dir,
    require_hf_datasets,
    save_raw_jsonl,
)

DATASET_NAME = "temporalwiki"

# Each entry: (period_id, hf_dataset_id, hf_config_or_None, split)
PERIOD_SOURCES = [
    ("period_2022", "seonghyeonye/TemporalWiki", None,           "train"),
    ("period_2023", "wikimedia/wikipedia",        "20231101.en", "train"),
]


def _stream_period(hf, hf_id: str, config, split: str,
                   period: str, n_docs: int):
    """Stream articles from HF and return a list of raw row dicts."""
    kwargs = {"streaming": True}
    if config:
        kwargs["name"] = config
    ds = hf.load_dataset(hf_id, split=split, **kwargs)

    raw_rows = []
    seen: set = set()

    for ex in ds:
        text = (ex.get("text") or ex.get("content") or "").strip()
        if len(text) < 200 or text[:50] in seen:
            continue
        seen.add(text[:50])
        raw_rows.append({
            "period": period,
            "title":  ex.get("title", ""),
            "text":   text,
        })
        if len(raw_rows) >= n_docs:
            break

    return raw_rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download TemporalWiki raw JSONL files from HuggingFace."
    )
    p.add_argument("--max_periods",        type=int, default=0,
                   help="Max number of periods to download (0 = all).")
    p.add_argument("--max_docs_per_period", type=int, default=0,
                   help="Max articles per period (0 = no cap).")
    args = p.parse_args()

    n_periods = cap(args.max_periods)
    n_docs    = cap(args.max_docs_per_period)

    hf  = require_hf_datasets()
    raw = raw_dir(DATASET_NAME)

    periods = (PERIOD_SOURCES if n_periods >= len(PERIOD_SOURCES)
               else PERIOD_SOURCES[:n_periods])

    for (pid, hf_id, config, split) in periods:
        print(f"[download:temporalwiki] streaming {pid} from {hf_id!r} …")
        raw_rows = _stream_period(hf, hf_id, config, split, pid, n_docs)
        save_raw_jsonl(raw_rows, raw, f"{pid}.jsonl")
        print(f"  · {pid}: {len(raw_rows):,} articles saved")

    print(f"[download:temporalwiki] done → {raw}")
    print(f"[download:temporalwiki] next step: "
          f"python Phase0/data/preprocess_temporalwiki.py")


if __name__ == "__main__":
    main()
