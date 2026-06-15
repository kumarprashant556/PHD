"""Build StreamingQA processed dataset from CC-News raw JSONL.

StreamingQA (Liska et al., 2022) is based on CC-News.  We derive temporal
open-QA probes from the same source corpus, giving us the same temporal
structure (monthly periods) without requiring the original dataset's
QA annotations.

Reads   : Phase0/data/raw/cc_news/raw.jsonl
            Each line: {"date": str, "title": str, "text": str, "url": str}
Writes  : Phase0/data/processed/streaming_qa/
            stream/<YYYY-MM>.jsonl  — one doc per article
            probes/<YYYY-MM>.jsonl  — one open-QA probe per article
            timeline.json
            metadata.json

Probe schema (unified CAPSEL format):
  {
    "question":   str,   # "According to the passage: <sentence with 'what entity'>?"
    "answer":     str,   # last proper noun in the sentence
    "choices":    {},    # empty — open-answer dataset
    "answer_key": "",    # empty — open-answer dataset
    "evidence":   str,   # source sentence
    "date":       str,   # article date string
    "period":     str,   # "YYYY-MM"
    "source":     str,   # "streaming_qa"
  }

Memory behaviour
----------------
CC-News raw.jsonl is ~1.6 GB.  This script reads it line-by-line without
ever loading the full file into RAM, and writes each period's output
directly to disk via open file handles.  It exits early once all requested
periods are filled on both docs and probes.

Default caps (safe for most machines):
  --max_periods 48          (≤ 4 years of monthly data)
  --max_docs_per_period 500
  --max_probes_per_period 500

Dependencies: none (pure stdlib)

Run::

    python Phase0/data/preprocess_streaming_qa.py
    python Phase0/data/preprocess_streaming_qa.py --max_periods 24 \\
        --max_docs_per_period 200 --max_probes_per_period 200
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import re
from collections import defaultdict
from typing import Any, Dict, IO, List, Tuple

from _utils import (
    cap,
    dataset_dir,
    iter_raw_jsonl,
    make_open_qa_probe,
    raw_dir,
    write_timeline,
)

DATASET_NAME   = "streaming_qa"
SOURCE_DATASET = "cc_news"

# Hard ceilings — overridden downward only, never upward, to protect RAM
_MAX_PERIODS_HARD  = 48
_MAX_DOCS_HARD     = 500
_MAX_PROBES_HARD   = 500


def _month_period(date_str: str) -> str:
    """'2017-03-21 ...' → '2017-03'.  Returns 'unknown' on bad input."""
    d = re.sub(r"[^0-9]", "", str(date_str or ""))
    if len(d) < 6:
        return "unknown"
    try:
        year, month = int(d[:4]), int(d[4:6])
    except ValueError:
        return "unknown"
    return f"{year:04d}-{month:02d}"


def _make_stream_doc(text: str, period: str, doc_idx: int,
                     title: str = "", url: str = "") -> Dict[str, Any]:
    return {
        "text":     text,
        "doc_id":   f"{DATASET_NAME}_{period}_{doc_idx:06d}",
        "period":   period,
        "source":   DATASET_NAME,
        "char_len": len(text),
        "title":    title,
        "url":      url,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build StreamingQA processed JSONL from CC-News raw data."
    )
    p.add_argument("--max_periods",           type=int, default=0,
                   help=f"Max monthly periods to output "
                        f"(0 = use hard cap of {_MAX_PERIODS_HARD}).")
    p.add_argument("--max_docs_per_period",   type=int, default=0,
                   help=f"Max stream docs per period "
                        f"(0 = use hard cap of {_MAX_DOCS_HARD}).")
    p.add_argument("--max_probes_per_period", type=int, default=0,
                   help=f"Max probes per period "
                        f"(0 = use hard cap of {_MAX_PROBES_HARD}).")
    p.add_argument("--force", action="store_true",
                   help="Re-preprocess even if processed output already exists.")
    args = p.parse_args()

    # Apply user caps, but never exceed the hard ceilings
    eff_periods = min(cap(args.max_periods),           _MAX_PERIODS_HARD)
    eff_docs    = min(cap(args.max_docs_per_period),   _MAX_DOCS_HARD)
    eff_probes  = min(cap(args.max_probes_per_period), _MAX_PROBES_HARD)

    out = dataset_dir(DATASET_NAME)
    if not args.force and (out / "timeline.json").exists():
        print(f"[preprocess:streaming_qa] already preprocessed — skipping "
              f"(--force to re-run)")
        return

    # ── Locate CC-News raw JSONL ──────────────────────────────────────────────
    cc_raw = raw_dir(SOURCE_DATASET) / "raw.jsonl"
    if not cc_raw.exists():
        raise SystemExit(
            f"[preprocess:streaming_qa] {cc_raw} not found.\n"
            f"Run first: python Phase0/data/download_streaming_qa.py"
        )

    print(f"[preprocess:streaming_qa] reading {cc_raw} …")
    print(f"  caps: ≤{eff_periods} periods, "
          f"≤{eff_docs} docs/period, ≤{eff_probes} probes/period")

    # ── Per-period state (streaming, no large in-memory dicts) ────────────────
    period_docs:    Dict[str, int] = defaultdict(int)
    period_probes:  Dict[str, int] = defaultdict(int)
    period_doc_idx: Dict[str, int] = defaultdict(int)
    period_order:   List[str]      = []                    # insertion-order
    fh_stream:      Dict[str, IO]  = {}
    fh_probes:      Dict[str, IO]  = {}

    def _open_period(pid: str) -> None:
        """Open output files for a new period."""
        fh_stream[pid] = open(out / "stream" / f"{pid}.jsonl", "w",
                              encoding="utf-8")
        fh_probes[pid] = open(out / "probes" / f"{pid}.jsonl", "w",
                              encoding="utf-8")
        period_order.append(pid)

    def _write_doc(pid: str, doc: Dict) -> None:
        fh_stream[pid].write(json.dumps(doc, ensure_ascii=False) + "\n")
        period_docs[pid] += 1

    def _write_probe(pid: str, probe: Dict) -> None:
        fh_probes[pid].write(json.dumps(probe, ensure_ascii=False) + "\n")
        period_probes[pid] += 1

    def _close_all() -> None:
        for fh in list(fh_stream.values()) + list(fh_probes.values()):
            fh.close()

    # ── Stream CC-News line by line ───────────────────────────────────────────
    try:
        for ex in iter_raw_jsonl(cc_raw):
            date_str = str(ex.get("date") or "")
            pid      = _month_period(date_str)
            if pid == "unknown":
                continue

            # Skip new periods once we have enough
            is_new = pid not in fh_stream
            if is_new and len(period_order) >= eff_periods:
                continue

            text = (ex.get("text") or "").strip()
            if len(text) < 80:
                continue

            docs_full   = period_docs[pid]   >= eff_docs
            probes_full = period_probes[pid] >= eff_probes
            if docs_full and probes_full:
                continue

            if is_new:
                _open_period(pid)

            # Stream doc
            if not docs_full:
                doc = _make_stream_doc(
                    text, period=pid,
                    doc_idx=period_doc_idx[pid],
                    title=ex.get("title", ""),
                    url=ex.get("url", ""),
                )
                _write_doc(pid, doc)
                period_doc_idx[pid] += 1

            # Open-QA probe
            if not probes_full:
                pr = make_open_qa_probe(
                    text, period=pid, source=DATASET_NAME, date=date_str
                )
                if pr is not None:
                    _write_probe(pid, pr)

            # Early exit: all desired periods filled on both docs and probes
            if len(period_order) >= eff_periods:
                if all(period_docs[p]   >= eff_docs and
                       period_probes[p] >= eff_probes
                       for p in period_order):
                    print("  [early exit] all periods filled.")
                    break

    finally:
        _close_all()

    # ── Timeline & metadata ───────────────────────────────────────────────────
    timeline = sorted(period_order)
    counts: List[Tuple] = [
        (pid, period_docs[pid], period_probes[pid]) for pid in timeline
    ]
    for pid, nd, np_ in counts:
        print(f"  · {pid}: {nd} docs, {np_} probes")

    write_timeline(out, timeline, {
        "source":             str(cc_raw),
        "note":               ("Derived from CC-News; same source domain as "
                               "umilossegura/streamingqa"),
        "period_granularity": "month",
        "probe_format":       "open_qa",
        "counts":             counts,
    })
    print(f"[preprocess:streaming_qa] done → {out}")


if __name__ == "__main__":
    main()
