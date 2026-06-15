"""TemporalWiki period loader  (data/temporalwiki.py)

Source     : datasets/temporalwiki/processed/stream/<period>.jsonl
Coverage   : Wikipedia article snapshots at different points in time
Period key : matches processed filename stem (e.g. "period_2022")
Task       : seq2seq text completion  (NO QA/MCQ)

Schema (each line in the JSONL files):
    {"text": str, "doc_id": str, "period": str,
     "source": "temporalwiki", "char_len": int}

Completion framing is applied at load-time via _base.finalise():
    input_text  = "complete: " + first split_frac of text
    target_text = next max_target_words words

Usage
-----
from data.temporalwiki import load_temporalwiki_periods

periods = load_temporalwiki_periods(n_per_period=10_000)
# {"period_2022": Dataset(input_text, target_text, period), ...}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset

from ._base import finalise, clean_text

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROCESSED_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets" / "temporalwiki" / "processed" / "stream"
)


def _list_periods() -> List[str]:
    if not _PROCESSED_DIR.exists():
        return []
    return sorted(
        p.stem for p in sorted(_PROCESSED_DIR.glob("*.jsonl"))
        if p.stat().st_size > 0
    )


def _load_jsonl(path: Path, n: int, seed: int) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(rows) > n:
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def load_temporalwiki_periods(
    periods: Optional[List[str]] = None,
    n_per_period: int = 10_000,
    split_frac: float = 0.50,
    max_target_words: int = 200,
    seed: int = 42,
    num_proc: int = 4,
    max_periods: Optional[int] = None,
) -> Dict[str, Dataset]:
    """Load TemporalWiki as temporal text-completion periods.

    Parameters
    ----------
    periods        : explicit list of period keys to load; None = all
    n_per_period   : max articles per period
    split_frac     : completion split fraction (encoder gets first half)
    max_target_words: decoder target cap (words)
    seed           : random seed for subsampling
    num_proc       : parallel workers (passed to finalise)
    max_periods    : cap on number of periods loaded (chronological order)

    Returns
    -------
    Dict[period_id, Dataset]  — columns: input_text, target_text, period
    """
    available = _list_periods()
    if not available:
        raise FileNotFoundError(
            f"No processed TemporalWiki stream files found in:\n  {_PROCESSED_DIR}\n"
            "Run the preprocessing script first."
        )

    if periods is None:
        periods = available
    else:
        periods = [p for p in periods if p in available]

    if max_periods is not None:
        periods = periods[:max_periods]

    result: Dict[str, Dataset] = {}
    for period_label in periods:
        jsonl_path = _PROCESSED_DIR / f"{period_label}.jsonl"
        rows = _load_jsonl(jsonl_path, n_per_period * 3, seed)

        items = []
        for row in rows:
            text = clean_text(row.get("text", "") or "")
            if len(text.split()) >= 30:
                items.append({"text": text})

        if not items:
            print(f"  {period_label}: empty after filtering — skipping")
            continue

        ds = Dataset.from_list(items)
        ds = finalise(
            ds, period=period_label, seed=seed, n=n_per_period,
            split_frac=split_frac, max_target_words=max_target_words,
            text_col="text", num_proc=num_proc,
        )
        result[period_label] = ds
        print(f"  {period_label}: {len(ds):,} articles ready")

    return result
