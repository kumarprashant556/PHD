"""Shared utilities for all CAPSEL dataset loaders  (data/_base.py).

Every loader in this package follows the same contract:

    load_<name>_periods(
        periods      = None,        # list[str] — use dataset default if None
        n_per_period = 20_000,      # max items to keep per period
        split_frac   = 0.50,        # completion split: encoder gets first half
        seed         = 42,
        **kwargs                    # dataset-specific extras
    ) -> Dict[str, datasets.Dataset]

Each returned Dataset has exactly three columns:
    input_text   str   "complete: " + first-half of document
    target_text  str   second-half of document (up to max_target_words)
    period       str   period identifier  e.g. "2017_H1", "2020-06", "gsm8k"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── HuggingFace datasets import guard ────────────────────────────────────────
# Callers often do sys.path.insert(0, REPO_ROOT) before importing this module.
# That places the local datasets/ data directory on sys.path before installed
# packages, so `import datasets` silently resolves to the wrong location.
# Strip repo-root and CWD entries for this import only, then restore.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_sp_backup = sys.path[:]
sys.path = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)]
from datasets import Dataset
sys.path[:] = _sp_backup
del _sp_backup, _REPO_ROOT


# ── Completion framing ────────────────────────────────────────────────────────

COMPLETION_PREFIX = "complete: "
MIN_WORDS         = 30    # articles shorter than this are dropped
MAX_TARGET_WORDS  = 200   # hard cap on target length (words, not tokens)


def apply_completion(
    text: str,
    split_frac: float = 0.50,
    max_target_words: int = MAX_TARGET_WORDS,
    prefix: str = COMPLETION_PREFIX,
) -> Dict[str, str]:
    """Split *text* into encoder input and decoder target.

    Returns {"input_text": ..., "target_text": ...} or empty strings
    if the document is too short.
    """
    words = text.split()
    if len(words) < MIN_WORDS:
        return {"input_text": "", "target_text": ""}
    mid = max(MIN_WORDS // 2, int(len(words) * split_frac))
    return {
        "input_text":  prefix + " ".join(words[:mid]),
        "target_text": " ".join(words[mid: mid + max_target_words]),
    }


def apply_completion_batch(
    batch: Dict[str, list],
    text_col: str = "text",
    split_frac: float = 0.50,
    max_target_words: int = MAX_TARGET_WORDS,
) -> Dict[str, list]:
    """Batched version of apply_completion for Dataset.map(batched=True)."""
    inputs, targets = [], []
    for text in batch[text_col]:
        result = apply_completion(text or "", split_frac, max_target_words)
        inputs.append(result["input_text"])
        targets.append(result["target_text"])
    return {"input_text": inputs, "target_text": targets}


# ── Dataset utilities ─────────────────────────────────────────────────────────

def subsample(ds: Dataset, n: int, seed: int = 42) -> Dataset:
    """Shuffle and keep at most *n* examples."""
    ds = ds.shuffle(seed=seed)
    if len(ds) > n:
        ds = ds.select(range(n))
    return ds


def drop_short(ds: Dataset, col: str = "input_text", min_len: int = 20) -> Dataset:
    """Drop examples where *col* is too short (empty / very short articles)."""
    return ds.filter(lambda x: len(x[col]) >= min_len)


def keep_columns(ds: Dataset, cols: List[str]) -> Dataset:
    """Remove all columns except *cols*."""
    to_remove = [c for c in ds.column_names if c not in cols]
    if to_remove:
        ds = ds.remove_columns(to_remove)
    return ds


STANDARD_COLS = ["input_text", "target_text", "period"]


def finalise(ds: Dataset, period: str, seed: int, n: int,
             split_frac: float, max_target_words: int,
             text_col: str = "text", num_proc: int = 4) -> Dataset:
    """Apply completion framing, subsample, drop short docs, standardise columns.

    This is the shared post-processing pipeline called by every loader.
    """
    # Apply completion split
    ds = ds.map(
        lambda batch: apply_completion_batch(batch, text_col, split_frac, max_target_words),
        batched=True,
        batch_size=1024,
        num_proc=num_proc,
        desc=f"  completion split ({period})",
    )
    # Inject period label if not present
    if "period" not in ds.column_names:
        ds = ds.map(lambda _: {"period": period}, num_proc=num_proc)
    else:
        ds = ds.map(lambda _: {"period": period}, num_proc=num_proc)

    # Drop empty / too-short docs
    ds = drop_short(ds, col="input_text", min_len=len(COMPLETION_PREFIX) + 20)
    ds = drop_short(ds, col="target_text", min_len=10)

    # Subsample
    ds = subsample(ds, n, seed)

    # Keep only standard columns
    ds = keep_columns(ds, STANDARD_COLS)
    return ds


# ── Text cleaning ─────────────────────────────────────────────────────────────

_WS_RE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """Collapse whitespace, strip leading/trailing."""
    return _WS_RE.sub(' ', text).strip()
