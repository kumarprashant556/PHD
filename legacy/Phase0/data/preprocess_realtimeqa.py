"""Preprocess raw RealtimeQA Parquet into CAPSEL period JSONL.

Reads   : Phase0/data/raw/realtimeqa/train.parquet   (requires pyarrow)
Writes  : Phase0/data/processed/realtimeqa/
            stream/<period>.jsonl   — one doc per evidence passage
            probes/<period>.jsonl   — one probe per question
            timeline.json
            metadata.json

Probe schema (unified CAPSEL format):
  {
    "question":   str,   # natural-language question
    "answer":     str,   # correct answer text (seq2seq target, Track A)
    "choices":    dict,  # {"A": ..., "B": ..., ...}  (MCQ, Track B)
    "answer_key": str,   # "A"/"B"/...                (MCQ, Track B)
    "evidence":   str,   # supporting passage (may be empty)
    "date":       str,   # ISO date string
    "period":     str,   # e.g. "2023-W01"
    "source":     str,   # "realtimeqa"
  }

Dependencies: pip install pyarrow

Run::

    python Phase0/data/preprocess_realtimeqa.py
    python Phase0/data/preprocess_realtimeqa.py --max_docs_per_period 200
    python Phase0/data/preprocess_realtimeqa.py --max_periods 8
    python Phase0/data/preprocess_realtimeqa.py --force
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import ast
import datetime
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from _utils import (
    cap,
    dataset_dir,
    make_doc,
    raw_dir,
    write_jsonl,
    write_timeline,
)

DATASET_NAME = "realtimeqa"


# ── Parsers (identical logic to original download_realtimeqa.py) ───────────

def _parse_choices(raw) -> Dict[str, str]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {k: v for k, v in zip("ABCD", raw)}
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return {k: v for k, v in zip("ABCD", parsed)}
        except Exception:
            pass
    return {}


def _parse_answer_key(raw) -> str:
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, int) or (isinstance(raw, str) and str(raw).isdigit()):
        idx = int(raw)
        return "ABCD"[idx] if 0 <= idx < 4 else "A"
    key = str(raw)[:1].upper()
    return key if key in "ABCD" else "A"


def _period_id(ex: Dict) -> str:
    """Map a row's date field to 'YYYY-Www' ISO week period id."""
    raw = str(ex.get("question_date") or ex.get("date") or "")
    d   = re.sub(r"[^0-9]", "", raw)
    if not re.match(r"^\d{8}$", d):
        return "unknown"
    try:
        y, m, day = int(d[:4]), int(d[4:6]), int(d[6:8])
        iso = datetime.date(y, m, day).isocalendar()
        return f"{iso[0]:04d}-W{iso[1]:02d}"
    except ValueError:
        return "unknown"


def _to_probe(ex: Dict, period: str) -> Dict:
    """Convert a raw row into the unified CAPSEL probe dict."""
    q          = (ex.get("question_sentence") or ex.get("question") or "").strip()
    choices    = _parse_choices(ex.get("choices"))
    key        = _parse_answer_key(ex.get("answer") or ex.get("correct_answer"))
    answer_txt = choices.get(key, "")
    evidence   = ex.get("evidence") or ex.get("context") or ""
    if isinstance(evidence, list):
        evidence = " ".join(str(e) for e in evidence)
    date = (ex.get("question_date") or ex.get("date") or "").strip()
    return {
        "question":   q,
        "answer":     answer_txt,
        "choices":    choices,
        "answer_key": key,
        "evidence":   str(evidence)[:500],
        "date":       date,
        "period":     period,
        "source":     DATASET_NAME,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess raw RealtimeQA Parquet → CAPSEL period JSONL."
    )
    p.add_argument("--max_periods",         type=int, default=0,
                   help="Max periods to output (0 = all).")
    p.add_argument("--max_docs_per_period", type=int, default=0,
                   help="Max stream docs per period (0 = all).")
    p.add_argument("--max_probes_per_period", type=int, default=0,
                   help="Max probes per period (0 = all).")
    p.add_argument("--force", action="store_true",
                   help="Re-preprocess even if processed output already exists.")
    args = p.parse_args()

    n_periods = cap(args.max_periods)
    n_docs    = cap(args.max_docs_per_period)
    n_probes  = cap(args.max_probes_per_period)

    # ── Locate raw Parquet ────────────────────────────────────────────────────
    raw     = raw_dir(DATASET_NAME)
    parquet = raw / "train.parquet"
    if not parquet.exists():
        raise SystemExit(
            f"[preprocess:realtimeqa] {parquet} not found.\n"
            f"Run first: python Phase0/data/download_realtimeqa.py"
        )

    # ── Load with pyarrow ─────────────────────────────────────────────────────
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        raise SystemExit(
            "[preprocess:realtimeqa] pyarrow is required to read Parquet.\n"
            "Install with:  pip install pyarrow"
        )

    print(f"[preprocess:realtimeqa] reading {parquet} …")
    table     = pq.read_table(str(parquet))
    col_dict  = table.to_pydict()
    col_names = list(col_dict.keys())
    n_rows    = table.num_rows
    print(f"  {n_rows:,} rows | columns: {col_names}")

    # ── Bucket rows by period ─────────────────────────────────────────────────
    by_period: Dict[str, Dict] = defaultdict(
        lambda: {"docs": [], "probes": [], "_idx": [0]}
    )

    for i in range(n_rows):
        ex  = {k: col_dict[k][i] for k in col_names}
        pid = _period_id(ex)
        if pid == "unknown":
            continue

        bucket = by_period[pid]

        # Stream doc — use evidence passage as text
        evidence = ex.get("evidence") or ex.get("context") or ""
        if isinstance(evidence, list):
            evidence = " ".join(str(e) for e in evidence)
        ev_str = str(evidence).strip()

        if ev_str and len(ev_str) > 80 and len(bucket["docs"]) < n_docs:
            idx = bucket["_idx"][0]
            bucket["_idx"][0] += 1
            bucket["docs"].append(
                make_doc(ev_str, period=pid, source=DATASET_NAME, doc_idx=idx)
            )

        # Probe
        if len(bucket["probes"]) < n_probes:
            probe = _to_probe(ex, pid)
            if probe["question"] and probe["answer"]:
                bucket["probes"].append(probe)

    # ── Force check ───────────────────────────────────────────────────────────
    out = dataset_dir(DATASET_NAME)
    if not args.force and (out / "timeline.json").exists():
        print(f"[preprocess:realtimeqa] already preprocessed — skipping "
              f"(--force to re-run)")
        return

    # ── Write output ──────────────────────────────────────────────────────────
    timeline = sorted(by_period)
    if n_periods < len(timeline):
        timeline = timeline[:n_periods]

    counts: List[Tuple] = []

    for pid in timeline:
        d = by_period[pid]
        write_jsonl(out / "stream" / f"{pid}.jsonl", d["docs"])
        write_jsonl(out / "probes" / f"{pid}.jsonl", d["probes"])
        counts.append((pid, len(d["docs"]), len(d["probes"])))
        print(f"  · {pid}: {len(d['docs'])} docs, {len(d['probes'])} probes")

    write_timeline(out, timeline, {
        "source":        "prajaktakini/realtime_qa",
        "period_scheme": "year-week",
        "probe_format":  "open_answer_mc",
        "counts":        counts,
    })
    print(f"[preprocess:realtimeqa] done → {out}")


if __name__ == "__main__":
    main()
