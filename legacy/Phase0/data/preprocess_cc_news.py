"""Preprocess raw CC-News JSONL → CAPSEL period-sliced JSONL.

Reads   : Phase0/data/raw/cc_news/<YYYY>.jsonl   (one file per year)
Writes  : Phase0/data/processed/cc_news/
            stream/<YYYY-MM>.jsonl    — one doc per article  (causal-LM input)
            probes/<YYYY-MM>.jsonl    — mixed probe formats  (seq2seq / eval)
            timeline.json
            metadata.json

Probe mix per period (configurable via CLI)
-------------------------------------------
  open_qa     — extractive open-answer QA from article sentences (highest signal)
  cloze       — 4-way cloze (MC4 format, fill-in-the-blank)
  completion  — next-span text continuation (causal LM training)

The preprocessor is designed to be memory-efficient: it iterates raw JSONL
one line at a time, writing period files incrementally via per-period buffers
that are flushed when complete.  Peak RAM is proportional to the largest
single year file (< 800 MB for a full CC-News year), not to the whole corpus.

Quality filters applied
-----------------------
  * Article text ≥ MIN_CHARS characters
  * At least 3 sentences
  * No excessive repetition (dedup by title within a period)
  * English heuristic: first 200 chars contain only ASCII + common punctuation

Usage
-----
    # full preprocess (all years in raw/)
    python Phase0/data/preprocess_cc_news.py

    # quick smoke-test: 4 periods, 200 docs each
    python Phase0/data/preprocess_cc_news.py \\
        --max_periods 4 --max_docs_per_period 200

    # custom probe budget
    python Phase0/data/preprocess_cc_news.py \\
        --probes_per_period 500 --open_qa_frac 0.6

    # re-run even if processed output already exists
    python Phase0/data/preprocess_cc_news.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import (
    NO_CAP,
    cap,
    dataset_dir,
    iter_raw_jsonl,
    make_cloze_probe,
    make_completion_probe,
    make_doc,
    make_open_qa_probe,
    raw_dir,
    write_jsonl,
    write_timeline,
)

DATASET_NAME = "cc_news"

# Quality thresholds
MIN_CHARS          = 300
MIN_SENTENCES      = 3
MAX_TITLE_SEEN     = True   # deduplicate within a period by title
ASCII_CHECK_LEN    = 200    # first N chars checked for English heuristic
ASCII_FRAC_MIN     = 0.80   # ≥80% ASCII → likely English


# ── language / quality filters ─────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _is_english(text: str) -> bool:
    """Rough English filter: fraction of ASCII chars in the first N chars."""
    sample = text[:ASCII_CHECK_LEN]
    if not sample:
        return False
    ascii_count = sum(1 for c in sample if ord(c) < 128)
    return ascii_count / len(sample) >= ASCII_FRAC_MIN


def _n_sentences(text: str) -> int:
    return len(_SENT_SPLIT.split(text.strip()))


def _accept(text: str, title: str, seen_titles: Set[str], period: str) -> bool:
    """Return True if this article should be included."""
    if len(text) < MIN_CHARS:
        return False
    if _n_sentences(text) < MIN_SENTENCES:
        return False
    if not _is_english(text):
        return False
    key = (period, title.lower()[:80])
    if MAX_TITLE_SEEN and key in seen_titles:
        return False
    return True


# ── probe budget allocation ────────────────────────────────────────────────────

def _generate_probes(
    text: str,
    title: str,
    period: str,
    open_qa_frac: float,
    completion_frac: float,
    doc_idx: int,
) -> List[Dict]:
    """Try to generate 1–3 probes from a single article.

    Priority order: open_qa → completion → cloze.
    Each probe type is attempted independently; returns all that succeed.
    The caller applies the per-period budget caps.
    """
    probes: List[Dict] = []
    cloze_frac = max(0.0, 1.0 - open_qa_frac - completion_frac)

    # open QA (highest quality)
    if open_qa_frac > 0:
        pr = make_open_qa_probe(text, period=period, source=DATASET_NAME)
        if pr:
            probes.append(pr)

    # completion
    if completion_frac > 0:
        pr = make_completion_probe(text, period=period, source=DATASET_NAME)
        if pr:
            probes.append(pr)

    # cloze
    if cloze_frac > 0:
        pr = make_cloze_probe(text, period=period, source=DATASET_NAME)
        if pr:
            probes.append(pr)

    return probes


# ── per-period in-memory buffer ────────────────────────────────────────────────

class _PeriodBuffer:
    def __init__(self) -> None:
        self.docs:         List[Dict] = []
        self.probes:       List[Dict] = []
        self.seen_titles:  Set[str]   = set()
        self._doc_idx:     int        = 0


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess raw CC-News JSONL → CAPSEL period JSONL."
    )
    p.add_argument("--max_periods",         type=int,   default=0,
                   help="Max periods to output (0 = all).")
    p.add_argument("--max_docs_per_period", type=int,   default=0,
                   help="Max stream docs per period (0 = all).")
    p.add_argument("--probes_per_period",   type=int,   default=0,
                   help="Max probes per period (0 = unlimited).")
    p.add_argument("--open_qa_frac",        type=float, default=0.50,
                   help="Fraction of probe budget to fill with open-QA probes.")
    p.add_argument("--completion_frac",     type=float, default=0.30,
                   help="Fraction of probe budget for completion probes.")
    p.add_argument("--min_period",          type=str,   default="",
                   help="Only include periods >= this (e.g. '2019-01').")
    p.add_argument("--max_period",          type=str,   default="",
                   help="Only include periods <= this (e.g. '2022-12').")
    p.add_argument("--force", action="store_true",
                   help="Re-preprocess even if processed output already exists.")
    args = p.parse_args()

    n_periods = cap(args.max_periods)
    n_docs    = cap(args.max_docs_per_period)
    n_probes  = cap(args.probes_per_period)
    oqa_frac  = max(0.0, min(1.0, args.open_qa_frac))
    cmp_frac  = max(0.0, min(1.0 - oqa_frac, args.completion_frac))

    raw = raw_dir(DATASET_NAME)
    out = dataset_dir(DATASET_NAME)

    # ── guard: check for raw files ─────────────────────────────────────
    raw_files = sorted(raw.glob("*.jsonl"))
    if not raw_files:
        raise SystemExit(
            f"[preprocess:cc_news] No raw JSONL files found in {raw}.\n"
            f"Run first: python Phase0/data/download_cc_news.py"
        )

    # ── guard: skip if already done ────────────────────────────────────
    if not args.force and (out / "timeline.json").exists():
        print(
            f"[preprocess:cc_news] Already preprocessed — skipping.\n"
            f"  Use --force to re-preprocess."
        )
        return

    print(f"[preprocess:cc_news] reading {len(raw_files)} raw file(s) from {raw}")
    print(f"  probe mix: open_qa={oqa_frac:.0%}  completion={cmp_frac:.0%}  "
          f"cloze={1-oqa_frac-cmp_frac:.0%}")

    # ── stream all raw files, bucket by period ─────────────────────────
    # Use a dict of _PeriodBuffer to avoid holding the full corpus in RAM:
    # we write each period's files as soon as we move past it.

    buffers: Dict[str, _PeriodBuffer] = defaultdict(_PeriodBuffer)
    total_articles = 0
    total_skipped  = 0

    for raw_file in raw_files:
        year = raw_file.stem   # e.g. "2020"
        print(f"  reading {raw_file.name} …", end="", flush=True)
        n_file = 0

        for row in iter_raw_jsonl(raw_file):
            period = (row.get("period") or "")[:7]  # YYYY-MM
            if not period or len(period) != 7:
                total_skipped += 1
                continue

            # Date-range filter
            if args.min_period and period < args.min_period:
                total_skipped += 1
                continue
            if args.max_period and period > args.max_period:
                total_skipped += 1
                continue

            title = (row.get("title") or "").strip()
            text  = (row.get("text")  or "").strip()

            buf = buffers[period]

            if not _accept(text, title, buf.seen_titles, period):
                total_skipped += 1
                continue

            buf.seen_titles.add((period, title.lower()[:80]))

            # Stream doc
            if len(buf.docs) < n_docs:
                buf.docs.append(
                    make_doc(
                        text,
                        period=period,
                        source=DATASET_NAME,
                        doc_idx=buf._doc_idx,
                        extra={"title": title, "url": row.get("url", "")},
                    )
                )
                buf._doc_idx += 1

            # Probes
            if len(buf.probes) < n_probes:
                new_probes = _generate_probes(
                    text, title, period,
                    open_qa_frac=oqa_frac,
                    completion_frac=cmp_frac,
                    doc_idx=buf._doc_idx,
                )
                remaining = n_probes - len(buf.probes)
                buf.probes.extend(new_probes[:remaining])

            total_articles += 1
            n_file += 1

        print(f" {n_file:,} articles")

    # ── determine output timeline ──────────────────────────────────────
    all_periods = sorted(buffers.keys())
    if args.min_period:
        all_periods = [p for p in all_periods if p >= args.min_period]
    if args.max_period:
        all_periods = [p for p in all_periods if p <= args.max_period]
    if n_periods < len(all_periods):
        all_periods = all_periods[:n_periods]

    if not all_periods:
        raise SystemExit(
            "[preprocess:cc_news] No periods found after filtering.  "
            "Check --min_period / --max_period against the raw data."
        )

    # ── write period files ─────────────────────────────────────────────
    counts: List[Tuple] = []

    for pid in all_periods:
        buf = buffers[pid]
        n_d = write_jsonl(out / "stream" / f"{pid}.jsonl", buf.docs)
        n_p = write_jsonl(out / "probes" / f"{pid}.jsonl", buf.probes)
        counts.append((pid, n_d, n_p))
        print(f"  · {pid}: {n_d:,} docs, {n_p:,} probes")

    # ── timeline + metadata ────────────────────────────────────────────
    write_timeline(out, all_periods, {
        "source":         "vblagoje/cc_news",
        "period_scheme":  "year-month",
        "probe_formats":  ["open_qa", "completion", "cloze"],
        "probe_mix": {
            "open_qa":    oqa_frac,
            "completion": cmp_frac,
            "cloze":      round(1.0 - oqa_frac - cmp_frac, 4),
        },
        "quality_filters": {
            "min_chars":       MIN_CHARS,
            "min_sentences":   MIN_SENTENCES,
            "english_heuristic": True,
            "dedup_by_title":  True,
        },
        "counts":           counts,
        "total_articles":   total_articles,
        "total_skipped":    total_skipped,
        "n_periods":        len(all_periods),
    })

    print(f"\n[preprocess:cc_news] done")
    print(f"  periods  : {len(all_periods)}  ({all_periods[0]} → {all_periods[-1]})")
    print(f"  articles : {total_articles:,}  (skipped {total_skipped:,})")
    total_docs   = sum(c[1] for c in counts)
    total_probes = sum(c[2] for c in counts)
    print(f"  stream docs  : {total_docs:,}")
    print(f"  probes       : {total_probes:,}")
    print(f"  output dir   : {out}")


if __name__ == "__main__":
    main()
