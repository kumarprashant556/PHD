"""Download the full CC-News corpus and save as raw JSONL.

Source  : HuggingFace ``vblagoje/cc_news`` (the freely-available mirror;
          the canonical ``commoncrawl/cc_news`` is gated).
Raw out : Phase0/data/raw/cc_news/
            YYYY.jsonl          — one file per calendar year, one article per line
            progress.json       — checkpoint so interrupted runs can resume
            manifest.json       — final counts per year after a complete run

The download is streaming and memory-constant: articles are written to disk
immediately rather than accumulated in memory.  A ``progress.json``
checkpoint is updated every ``--checkpoint_every`` articles so a Ctrl-C or
crash can be resumed without re-downloading from the start.

Full corpus size: ~708 K articles, ~1.7 GB uncompressed JSONL.
Expected runtime: 20–60 min depending on connection speed.

Usage
-----
    # full download (recommended)
    python Phase0/data/download_cc_news.py

    # resume an interrupted run (skips already-written articles automatically)
    python Phase0/data/download_cc_news.py

    # small smoke-test: first 5 000 articles only
    python Phase0/data/download_cc_news.py --max_articles 5000

    # re-download from scratch (ignores checkpoint)
    python Phase0/data/download_cc_news.py --force

After downloading, run the preprocessor::

    python Phase0/data/preprocess_cc_news.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, IO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import NO_CAP, cap, raw_dir, require_hf_datasets

DATASET_NAME = "cc_news"
HF_REPO      = "vblagoje/cc_news"

# Article quality filters
MIN_TEXT_CHARS = 200       # discard very short articles
MIN_TITLE_CHARS = 5        # discard untitled stubs
CHECKPOINT_EVERY = 10_000  # flush progress every N articles


# ── helpers ────────────────────────────────────────────────────────────────────

def _year(date_str: str) -> str:
    """Return 'YYYY' from a date string, or '' if unparseable."""
    s = (date_str or "").strip()[:4]
    return s if s.isdigit() and 2000 <= int(s) <= 2035 else ""


def _clean(text: str) -> str:
    """Strip leading/trailing whitespace and collapse excessive blank lines."""
    import re
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _load_progress(raw: Path) -> Dict[str, int]:
    """Return {year: articles_written} from the checkpoint file."""
    p = raw / "progress.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_progress(raw: Path, progress: Dict[str, int]) -> None:
    (raw / "progress.json").write_text(json.dumps(progress, indent=2))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Stream-download CC-News and write raw JSONL by year."
    )
    p.add_argument("--max_articles", type=int, default=0,
                   help="Stop after this many articles (0 = all, for smoke tests).")
    p.add_argument("--checkpoint_every", type=int, default=CHECKPOINT_EVERY,
                   help="Flush progress checkpoint every N articles.")
    p.add_argument("--force", action="store_true",
                   help="Ignore existing checkpoint and re-download from scratch.")
    args = p.parse_args()

    limit    = cap(args.max_articles)
    ckpt_n   = max(1, args.checkpoint_every)

    hf  = require_hf_datasets()
    raw = raw_dir(DATASET_NAME)

    # ── checkpoint / resume ────────────────────────────────────────────
    progress: Dict[str, int] = {} if args.force else _load_progress(raw)
    total_already = sum(progress.values())
    if total_already:
        print(f"[cc_news] resuming — {total_already:,} articles already written.")
        print(f"          years seen so far: {sorted(progress)}")
    else:
        print(f"[cc_news] starting fresh download from {HF_REPO!r}")

    # Open one file handle per year (append mode for resume)
    handles: Dict[str, IO] = {}

    def _get_handle(year: str) -> IO:
        if year not in handles:
            path = raw / f"{year}.jsonl"
            mode = "a" if path.exists() and not args.force else "w"
            handles[year] = open(path, mode, encoding="utf-8")
        return handles[year]

    # ── stream ─────────────────────────────────────────────────────────
    print(f"[cc_news] loading {HF_REPO!r} (streaming) …")
    ds = hf.load_dataset(HF_REPO, split="train", streaming=True, trust_remote_code=True)

    total_seen = total_already  # counts articles streamed in this session + prior
    total_written = 0            # written in this session
    skipped_qual = 0
    skipped_date = 0
    year_counts: Dict[str, int] = defaultdict(int, progress)

    t0 = time.time()

    for ex in ds:
        if total_written + total_already >= limit:
            break

        # ── date / year ────────────────────────────────────────────────
        date_raw = (ex.get("date") or ex.get("published") or "").strip()
        year = _year(date_raw)
        if not year:
            skipped_date += 1
            continue

        # ── year-month for downstream use ──────────────────────────────
        ym = date_raw[:7] if len(date_raw) >= 7 and date_raw[4] == "-" else ""

        # ── quality filter ─────────────────────────────────────────────
        title = _clean(ex.get("title") or "")
        text  = _clean(ex.get("text")  or "")
        if len(text) < MIN_TEXT_CHARS or len(title) < MIN_TITLE_CHARS:
            skipped_qual += 1
            continue

        # ── resume: skip articles already counted in prior run ─────────
        # (We rely on the file being append-only and the count in progress.)
        # If --force was given, progress is empty and we rewrite everything.
        prior = progress.get(year, 0)
        if year_counts[year] < prior:
            year_counts[year] += 1
            total_written += 1
            continue  # article already in file from a previous run

        # ── write ──────────────────────────────────────────────────────
        row = {
            "title":   title,
            "text":    text,
            "date":    date_raw,
            "year":    year,
            "period":  ym,
            "url":     (ex.get("url") or "").strip(),
            "domain":  (ex.get("domain") or "").strip(),
        }
        fh = _get_handle(year)
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        year_counts[year] += 1
        total_written += 1
        total_seen    += 1

        # ── progress report ────────────────────────────────────────────
        if total_written % ckpt_n == 0:
            elapsed = time.time() - t0
            rate    = total_written / max(elapsed, 1)
            _save_progress(raw, dict(year_counts))
            print(
                f"  [{total_written:>7,} written | {total_seen:>7,} seen | "
                f"{rate:.0f} art/s | {elapsed/60:.1f} min]  "
                f"years: {dict(sorted(year_counts.items()))}"
            )

    # ── close file handles ─────────────────────────────────────────────
    for fh in handles.values():
        fh.close()

    # ── final checkpoint + manifest ────────────────────────────────────
    _save_progress(raw, dict(year_counts))

    manifest = {
        "source":         HF_REPO,
        "total_articles": sum(year_counts.values()),
        "by_year":        dict(sorted(year_counts.items())),
        "skipped_no_date": skipped_date,
        "skipped_quality": skipped_qual,
        "files":          [f"{y}.jsonl" for y in sorted(year_counts)],
    }
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    print(f"\n[cc_news] download complete in {elapsed/60:.1f} min")
    print(f"  total articles : {manifest['total_articles']:,}")
    print(f"  skipped (date) : {skipped_date:,}")
    print(f"  skipped (qual) : {skipped_qual:,}")
    print(f"  per-year counts: {manifest['by_year']}")
    print(f"  raw dir        : {raw}")
    print(f"\nNext step: python Phase0/data/preprocess_cc_news.py")


if __name__ == "__main__":
    main()
