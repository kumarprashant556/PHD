"""Data loaders for CC-News v2 (stream + probes) and the pre-tokenized Dataset.

The v2 layout lives under ``local_data/cc_news/processed/``:

    stream_v2/<period>.jsonl    — training items   ({input, target})
    probes_v2/<period>.jsonl    — eval probes      ({input, target, probe_type, ...})

Periods we actually train on (the 2019 buckets have <5 docs and are dropped
upstream by ``preprocessing.run``):

    2017_H1 · 2017_H2 · 2018_H1 · 2018_H2
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset as TorchDataset

# Repo paths.  ``_runtime/__init__.py`` already put repo root on sys.path,
# so this works whether the caller invoked us via the package or the shim.
_ROOT          = Path(__file__).resolve().parents[2]
_DATASETS_ROOT = _ROOT / "local_data"
_CC_STREAM_V2  = _DATASETS_ROOT / "cc_news" / "processed" / "stream_v2"
_CC_PROBES_V2  = _DATASETS_ROOT / "cc_news" / "processed" / "probes_v2"
_CC_GOOD_PERIODS = ["2017_H1", "2017_H2", "2018_H1", "2018_H2"]

from .logging_setup import LOGGER_NAME  # noqa: E402


# ── Stream / probe file loaders ───────────────────────────────────────────────

def load_cc_news_v2(
    n_per_period: int = 20_000,
    seed: int = 42,
    max_periods: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Read processed CC-News v2 stream JSONL files into a dict of raw items.

    Returns ``{period_id: [{input_text, target_text}, ...]}`` with at most
    ``n_per_period`` items per period, deterministically shuffled by ``seed``.
    """
    if not _CC_STREAM_V2.exists():
        raise FileNotFoundError(
            f"CC-News v2 stream not found at {_CC_STREAM_V2}.\n"
            "Run:  python preprocessing/run.py cc_news --force"
        )
    available = sorted(
        p.stem for p in _CC_STREAM_V2.glob("*.jsonl") if p.stat().st_size > 100
    )
    periods = [p for p in _CC_GOOD_PERIODS if p in available] or available
    if max_periods:
        periods = periods[:max_periods]

    rng = random.Random(seed)
    log = logging.getLogger(f"{LOGGER_NAME}.data")
    result: Dict[str, List[Dict[str, Any]]] = {}
    for period in periods:
        rows: List[Dict[str, Any]] = []
        with open(_CC_STREAM_V2 / f"{period}.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inp = (obj.get("input")  or "").strip()
                tgt = (obj.get("target") or "").strip()
                if inp and tgt:
                    rows.append({"input_text": inp, "target_text": tgt})
        rng.shuffle(rows)
        result[period] = rows[:n_per_period]
        log.info("CC-News v2  %s: %d examples", period, len(result[period]))
    return result


def load_probes(period_id: str) -> List[Dict[str, Any]]:
    """Read v2 probes for a single period.  Returns ``[]`` if the file is absent."""
    path = _CC_PROBES_V2 / f"{period_id}.jsonl"
    if not path.exists():
        return []
    probes: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                probes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return probes


# ── Pre-tokenized dataset ─────────────────────────────────────────────────────

class TokenizedDataset(TorchDataset):
    """Pre-tokenize all items once; emit per-item dicts of plain int lists.

    Padding is intentionally deferred to ``DataCollatorForSeq2Seq`` (dynamic
    per-batch).  Tokenizing once upfront eliminates per-item tokenizer overhead
    inside DataLoader workers, which was the dominant bottleneck previously.
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        tokenizer,
        max_input_len: int,
        max_target_len: int,
        chunk: int = 1024,
    ) -> None:
        pad_id = tokenizer.pad_token_id or 0
        self.data: List[Dict[str, List[int]]] = []
        for i in range(0, len(items), chunk):
            batch = items[i:i + chunk]
            enc = tokenizer(
                [x["input_text"]  for x in batch],
                truncation=True, max_length=max_input_len, padding=False,
            )
            dec = tokenizer(
                [x["target_text"] for x in batch],
                truncation=True, max_length=max_target_len, padding=False,
            )
            for j in range(len(batch)):
                labels = [t if t != pad_id else -100 for t in dec["input_ids"][j]]
                self.data.append({
                    "input_ids":      enc["input_ids"][j],
                    "attention_mask": enc["attention_mask"][j],
                    "labels":         labels,
                })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.data[idx]
