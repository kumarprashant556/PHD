"""RedPajama-V2 streaming loader  (data/redpajama.py)

Used for the E-ROUTE routing ablation (T1.7): 6 Common Crawl snapshots
as temporal periods, streamed to avoid materialising 30T tokens.

Source     : togethercomputer/RedPajama-Data-V2  (HuggingFace streaming)
Coverage   : 84 CC snapshots, 2014–2023, English
Period key : CC snapshot ID  e.g. "2023-06", "2022-49"

Default 6 snapshots (≈ one per year, 2018–2023):
    2023-06  →  Period 1  (most recent)
    2022-12  →  Period 2
    2021-43  →  Period 3
    2020-24  →  Period 4
    2019-33  →  Period 5
    2018-43  →  Period 6  (oldest)

Usage
-----
from data.redpajama import load_redpajama_periods

periods = load_redpajama_periods(n_per_period=20_000)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

from datasets import Dataset

from ._base import finalise, clean_text, STANDARD_COLS

# ── Default snapshot layout ───────────────────────────────────────────────────

DEFAULT_SNAPSHOTS: List[str] = [
    "2023-06",   # ~2023
    "2022-12",   # ~2022
    "2021-43",   # ~2021
    "2020-24",   # ~2020
    "2019-33",   # ~2019
    "2018-43",   # ~2018
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _stream_snapshot(snapshot: str, n: int, language: str = "en") -> List[dict]:
    """Stream at most *n* documents from a single CC snapshot."""
    from datasets import load_dataset as hf_load

    ds = hf_load(
        "togethercomputer/RedPajama-Data-V2",
        snapshots=[snapshot],
        languages=[language],
        name="default",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
    # RedPajama-V2 has a 'raw_content' field (the document text)
    items = []
    for item in itertools.islice(ds, n * 5):   # oversample to account for short docs
        text = item.get("raw_content", item.get("text", ""))
        if text and len(text.split()) >= 30:
            items.append({"text": clean_text(text), "period": snapshot})
        if len(items) >= n:
            break
    return items


# ── Public API ────────────────────────────────────────────────────────────────

def load_redpajama_periods(
    snapshots: Optional[List[str]] = None,
    n_per_period: int = 20_000,
    split_frac: float = 0.50,
    max_target_words: int = 200,
    seed: int = 42,
    language: str = "en",
    num_proc: int = 1,           # streaming → single-process materialise
) -> Dict[str, Dataset]:
    """Stream RedPajama-V2 snapshots and return as period-keyed Datasets.

    Parameters
    ----------
    snapshots      : CC snapshot IDs to use as periods (default: 6 snapshots 2018–2023)
    n_per_period   : max documents to materialise per snapshot
    split_frac     : completion split fraction (encoder/decoder)
    max_target_words: cap on decoder target word count
    seed           : random seed for final subsample
    language       : language filter (default: "en")
    num_proc       : workers for post-processing map (keep 1 for streaming)

    Returns
    -------
    Dict[snapshot_id, Dataset]  — columns: input_text, target_text, period
    """
    if snapshots is None:
        snapshots = DEFAULT_SNAPSHOTS

    result: Dict[str, Dataset] = {}
    for snapshot in snapshots:
        print(f"  Streaming RedPajama-V2 snapshot {snapshot} …")
        try:
            items = _stream_snapshot(snapshot, n=n_per_period, language=language)
        except Exception as e:
            print(f"  WARNING: failed to stream {snapshot}: {e} — skipping")
            continue

        if not items:
            print(f"  WARNING: 0 documents collected for {snapshot}")
            continue

        ds = Dataset.from_list(items)
        ds = finalise(
            ds, period=snapshot, seed=seed, n=n_per_period,
            split_frac=split_frac, max_target_words=max_target_words,
            text_col="text", num_proc=num_proc,
        )
        result[snapshot] = ds
        print(f"  {snapshot}: {len(ds):,} documents ready")

    return result
