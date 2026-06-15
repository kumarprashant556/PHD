"""Download TiC-LM proxy — daily slices of C4 (realnewslike) as temporal periods.

TiC-LM (Li et al, ACL 2025) tests how well language models adapt to temporal
drift in a large text corpus.  The original ``apple/TiC-LM`` requires
authentication and is 2.9T tokens of monthly C4.

Proxy strategy
--------------
``allenai/c4`` (realnewslike) was built from a single Common Crawl snapshot
(April 2019), so all documents share the same crawl month.  However, within
that snapshot the ``timestamp`` field carries **full dates** (YYYY-MM-DD),
reflecting the actual publication date of each news article.  Bucketing by
day instead of by month yields up to 30 distinct temporal periods from a
single scan of ~120k docs — more than enough for a meaningful CL benchmark.

  period_2019-04-01  →  news published 1 Apr 2019
  period_2019-04-02  →  news published 2 Apr 2019
  …

The scan budget is set to n_periods × n_docs × 20 (capped at 2 000 000)
so the script terminates predictably rather than streaming forever.

Raw output   → Phase0/data/raw/tic_lm/<YYYY-MM-DD>.jsonl
Processed    → Phase0/data/processed/tic_lm/stream/<period>.jsonl
                                            /probes/<period>.jsonl

Run::

    # 6 days, 3 000 docs/day — recommended default (~360 000 docs scanned)
    python Phase0/data/download_tic_lm.py --max_periods 6 --max_docs_per_period 3000

    # larger run for full experiments (up to ~4 000 docs/day available)
    python Phase0/data/download_tic_lm.py --max_periods 12 --max_docs_per_period 4000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from collections import defaultdict
from typing import Dict, List

from _utils import (
    cap,
    dataset_dir,
    make_cloze_probe,
    make_completion_probe,
    make_doc,
    raw_dir,
    require_hf_datasets,
    save_raw_jsonl,
    write_jsonl,
    write_timeline,
)

DATASET_NAME = "tic_lm"
DEFAULT_HF_ID = "allenai/c4"
DEFAULT_CONFIG = "realnewslike"

SCAN_MULTIPLIER = 20
SCAN_CAP = 2_000_000


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download TiC-LM proxy from allenai/c4 realnewslike (daily periods)."
    )
    p.add_argument("--hf_id", default=DEFAULT_HF_ID)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--max_periods", type=int, default=0,
                   help="0 = all days found in scan window")
    p.add_argument("--max_docs_per_period", type=int, default=0,
                   help="0 = unlimited per day (recommend capping, e.g. 500)")
    p.add_argument("--probes_per_period", type=int, default=0,
                   help="0 = all probes we can build per day")
    p.add_argument("--scan_budget", type=int, default=0,
                   help="Override max docs to scan. Default: "
                        "min(n_periods * n_docs * 20, 2_000_000)")
    args = p.parse_args()

    n_periods = cap(args.max_periods)
    n_docs    = cap(args.max_docs_per_period)
    n_probes  = cap(args.probes_per_period)

    if args.scan_budget > 0:
        scan_budget = args.scan_budget
    else:
        scan_budget = min(n_periods * n_docs * SCAN_MULTIPLIER, SCAN_CAP)

    os.environ.setdefault("HF_DATASETS_TIMEOUT", "60")

    hf = require_hf_datasets()
    raw = raw_dir(DATASET_NAME)
    out = dataset_dir(DATASET_NAME)

    print(f"[tic_lm] streaming {args.hf_id!r} config={args.config!r} "
          f"→ daily buckets (YYYY-MM-DD)…")
    print(f"[tic_lm] caps: max_periods={n_periods}, max_docs={n_docs}, "
          f"n_probes={n_probes}, scan_budget={scan_budget:,}")

    # Load without cache_dir — HF Arrow cache stays in ~/.cache/huggingface
    ds = hf.load_dataset(args.hf_id, args.config,
                         split="train", streaming=True)

    # ── Phase 1: scan budget docs, bucket by YYYY-MM-DD ──────────────────────
    by_day: Dict[str, List[Dict]] = defaultdict(list)
    total_seen = 0

    for ex in ds:
        total_seen += 1
        if total_seen % 100_000 == 0:
            top = sorted(by_day.items(), key=lambda x: -len(x[1]))[:4]
            top_str = ", ".join(f"{d}:{len(docs)}" for d, docs in top)
            print(f"  [tic_lm] scanned {total_seen:,} | "
                  f"days: {len(by_day)} | top: {top_str}")

        if total_seen >= scan_budget:
            break

        # C4 timestamp field: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS" etc.
        ts = str(ex.get("timestamp") or "")
        day = ts[:10]   # "YYYY-MM-DD"
        if len(day) != 10 or day[4] != "-" or day[7] != "-":
            continue
        # Sanity check: must be a valid date prefix
        if not day[:4].isdigit() or not day[5:7].isdigit() or not day[8:10].isdigit():
            continue

        text = (ex.get("text") or "").strip()
        if len(text) < 80:
            continue

        bucket = by_day[day]
        if len(bucket) < n_docs:
            bucket.append(make_doc(text, period=day, source=DATASET_NAME,
                                   doc_idx=len(bucket)))

    print(f"[tic_lm] scan complete: {total_seen:,} docs scanned, "
          f"{len(by_day)} distinct days found.")

    if not by_day:
        print("[tic_lm] WARNING: no documents collected — check that the "
              "dataset has a 'timestamp' field with YYYY-MM-DD format.")
        sys.exit(1)

    print(f"[tic_lm] date range: {min(by_day)} → {max(by_day)}")

    # ── Phase 2: pick top n_periods days by doc count, sort chronologically ──
    ranked = sorted(by_day.items(), key=lambda x: -len(x[1]))
    selected_days = sorted(day for day, _ in ranked[:n_periods])
    print(f"[tic_lm] selected {len(selected_days)} days:")
    for day in selected_days:
        print(f"  {day}: {len(by_day[day]):,} docs available → "
              f"will use {min(len(by_day[day]), n_docs):,}")

    # ── Phase 3: save raw, build stream + probes ──────────────────────────────
    timeline = [f"period_{d}" for d in selected_days]
    counts = []

    for day in selected_days:
        pid  = f"period_{day}"
        docs = by_day[day][:n_docs]

        # Raw backup
        raw_rows = [{"period": pid, "date": day, "text": d["text"],
                     "doc_id": d.get("doc_id", "")} for d in docs]
        save_raw_jsonl(raw_rows, raw, f"{day}.jsonl")

        # Probes
        mc4_budget  = max(1, n_probes // 2)
        comp_budget = n_probes - mc4_budget
        second_half = docs[max(1, len(docs) // 2):]
        first_half  = docs[:max(1, len(docs) // 2)]

        probes: List[Dict] = []
        for d in second_half:
            pr = make_cloze_probe(d["text"], period=pid, source=DATASET_NAME)
            if pr is not None:
                probes.append(pr)
            if len(probes) >= mc4_budget:
                break
        for d in first_half:
            if len(probes) >= mc4_budget + comp_budget:
                break
            cp = make_completion_probe(d["text"], period=pid, source=DATASET_NAME)
            if cp is not None:
                probes.append(cp)

        write_jsonl(out / "stream" / f"{pid}.jsonl", docs)
        write_jsonl(out / "probes" / f"{pid}.jsonl", probes)
        counts.append((pid, len(docs), len(probes)))
        print(f"  · {pid}: {len(docs):,} docs, {len(probes):,} probes")

    write_timeline(out, timeline, {
        "source": f"{args.hf_id} ({args.config})",
        "period_scheme": "day",
        "note": "daily bucketing within the April 2019 CC crawl snapshot",
        "probe_formats": ["mc4", "completion"],
        "counts": counts,
    })
    print(f"[tic_lm] done → {out}")


if __name__ == "__main__":
    main()
