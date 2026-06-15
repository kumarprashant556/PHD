"""Preprocess raw TemporalWiki JSONL into CAPSEL period JSONL.

Reads   : Phase0/data/raw/temporalwiki/<period>.jsonl
            Each line: {"period": str, "title": str, "text": str}
Writes  : Phase0/data/processed/temporalwiki/
            stream/<period>.jsonl   — one doc per article
            probes/<period>.jsonl   — one open-QA probe per article
            timeline.json
            metadata.json

Probe schema (unified CAPSEL format):
  {
    "question":   str,   # "According to the passage: <sentence with 'what entity'>?"
    "answer":     str,   # last proper noun in the sentence
    "choices":    {},    # empty — open-answer dataset
    "answer_key": "",    # empty — open-answer dataset
    "evidence":   str,   # source sentence
    "date":       "",    # not available for Wikipedia snapshots
    "period":     str,   # "period_2022" or "period_2023"
    "source":     str,   # "temporalwiki"
  }

Dependencies: none (pure stdlib)

Run::

    python Phase0/data/preprocess_temporalwiki.py
    python Phase0/data/preprocess_temporalwiki.py --max_docs_per_period 500
    python Phase0/data/preprocess_temporalwiki.py --max_probes_per_period 200
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from typing import List, Tuple

from _utils import (
    cap,
    dataset_dir,
    iter_raw_jsonl,
    make_doc,
    make_open_qa_probe,
    raw_dir,
    write_jsonl,
    write_timeline,
)

DATASET_NAME = "temporalwiki"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess raw TemporalWiki JSONL → CAPSEL period JSONL."
    )
    p.add_argument("--max_docs_per_period",   type=int, default=0,
                   help="Max stream docs per period (0 = all).")
    p.add_argument("--max_probes_per_period", type=int, default=0,
                   help="Max probes per period (0 = all).")
    p.add_argument("--force", action="store_true",
                   help="Re-preprocess even if processed output already exists.")
    args = p.parse_args()

    n_docs   = cap(args.max_docs_per_period)
    n_probes = cap(args.max_probes_per_period)

    out = dataset_dir(DATASET_NAME)
    if not args.force and (out / "timeline.json").exists():
        print(f"[preprocess:temporalwiki] already preprocessed — skipping "
              f"(--force to re-run)")
        return

    # ── Locate raw JSONL files ────────────────────────────────────────────────
    raw      = raw_dir(DATASET_NAME)
    raw_files = sorted(raw.glob("*.jsonl"))

    if not raw_files:
        raise SystemExit(
            f"[preprocess:temporalwiki] No JSONL files found in {raw}.\n"
            f"Run first: python Phase0/data/download_temporalwiki.py"
        )

    print(f"[preprocess:temporalwiki] found {len(raw_files)} raw file(s) in {raw}")

    timeline: List[str] = []
    counts:   List[Tuple] = []

    for raw_file in raw_files:
        pid = raw_file.stem      # e.g. "period_2022"
        docs   = []
        probes = []
        doc_idx = 0

        for row in iter_raw_jsonl(raw_file):
            text = (row.get("text") or "").strip()
            if not text:
                continue

            # Stream doc
            if len(docs) < n_docs:
                docs.append(
                    make_doc(text, period=pid, source=DATASET_NAME,
                             doc_idx=doc_idx,
                             extra={"title": row.get("title", "")})
                )
                doc_idx += 1

            # Open-QA probe
            if len(probes) < n_probes:
                pr = make_open_qa_probe(text, period=pid, source=DATASET_NAME)
                if pr is not None:
                    probes.append(pr)

            if len(docs) >= n_docs and len(probes) >= n_probes:
                break

        write_jsonl(out / "stream" / f"{pid}.jsonl", docs)
        write_jsonl(out / "probes" / f"{pid}.jsonl", probes)
        timeline.append(pid)
        counts.append((pid, len(docs), len(probes)))
        print(f"  · {pid}: {len(docs)} docs, {len(probes)} probes")

    write_timeline(out, timeline, {
        "source":        "seonghyeonye/TemporalWiki + wikimedia/wikipedia",
        "period_scheme": "snapshot",
        "probe_format":  "open_qa",
        "counts":        counts,
    })
    print(f"[preprocess:temporalwiki] done → {out}")


if __name__ == "__main__":
    main()
