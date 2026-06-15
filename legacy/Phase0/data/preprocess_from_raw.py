"""Re-process CAPSEL Phase-0 datasets from locally stored raw files.

No HuggingFace download is performed.  All data is read from:

  Phase0/data/raw/realtimeqa/train.parquet   → RealtimeQA
  Phase0/data/processed/temporalwiki/stream/ → TemporalWiki (raw JSONL not
                                               saved at download time; stream
                                               files are the source of truth)
  Phase0/data/raw/cc_news/raw.jsonl          → StreamingQA (CC-News articles
                                               sliced into monthly periods)

Output (overwrites previous processed files):

  Phase0/data/processed/<dataset>/stream/<period>.jsonl
  Phase0/data/processed/<dataset>/probes/<period>.jsonl
  Phase0/data/processed/<dataset>/timeline.json
  Phase0/data/processed/<dataset>/metadata.json

Unified probe schema (all datasets):

  {
    "question":   str,   # natural-language question
    "answer":     str,   # target string for seq2seq (Track A)
    "choices":    dict,  # {"A": ..., "B": ..., ...}  — empty for open-answer only
    "answer_key": str,   # "A"/"B"/...  — empty for open-answer only
    "evidence":   str,   # supporting passage (may be empty for RealtimeQA)
    "date":       str,   # ISO-style date string (may be empty)
    "period":     str,   # e.g. "2023-W01", "2017-03", "period_2022"
    "source":     str,   # dataset name
  }

Dependencies
------------
  pip install pyarrow          # required for RealtimeQA (reads .parquet)
  # pandas, datasets NOT required

Usage::

    python Phase0/data/preprocess_from_raw.py
    python Phase0/data/preprocess_from_raw.py --datasets realtimeqa
    python Phase0/data/preprocess_from_raw.py --max_docs_per_period 500 \\
        --max_probes_per_period 500 --max_periods 24
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent   # Phase0/data/
RAW_ROOT  = HERE / "raw"
PROC_ROOT = HERE / "processed"

NO_CAP = 10 ** 12


def _cap(v: int) -> int:
    return NO_CAP if not v else int(v)


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yield dicts from a JSONL file one at a time (memory-efficient)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_timeline(out: Path, period_ids: List[str],
                    meta_extra: Optional[Dict] = None) -> None:
    (out / "timeline.json").write_text(json.dumps(period_ids, indent=2))
    md: Dict[str, Any] = {"timeline": period_ids, "n_periods": len(period_ids)}
    if meta_extra:
        md.update(meta_extra)
    (out / "metadata.json").write_text(json.dumps(md, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# Shared probe / doc builders
# ══════════════════════════════════════════════════════════════════════════════

def _make_open_qa_probe(text: str, period: str = "", source: str = "",
                        date: str = "") -> Optional[Dict]:
    """Extract a factual sentence with a proper noun and build an open-QA probe.

    Identical logic to _utils.make_open_qa_probe so processed output is
    consistent whether this script or the original downloader is used.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        if len(sent) < 40:
            continue
        nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", sent)
        if len(nouns) < 2:
            continue
        answer = nouns[-1]
        question = re.sub(
            rf"\b{re.escape(answer)}\b", "what entity",
            sent, count=1, flags=re.IGNORECASE,
        )
        question = f"According to the passage: {question}?"
        return {
            "question":   question.strip(),
            "answer":     answer.strip(),
            "choices":    {},
            "answer_key": "",
            "evidence":   sent.strip(),
            "date":       date,
            "period":     period,
            "source":     source,
        }
    return None


def _make_stream_doc(text: str, period: str, source: str,
                     doc_idx: int, extra: Optional[Dict] = None) -> Dict:
    parts  = [p for p in [source, period] if p]
    doc_id = "_".join(parts + [f"{doc_idx:06d}"]) if parts else f"doc_{doc_idx:06d}"
    doc: Dict[str, Any] = {
        "text":     text,
        "doc_id":   doc_id,
        "period":   period,
        "source":   source,
        "char_len": len(text),
    }
    if extra:
        doc.update(extra)
    return doc


# ══════════════════════════════════════════════════════════════════════════════
# RealtimeQA parsers  (same logic as download_realtimeqa._parse_* / _to_probe)
# ══════════════════════════════════════════════════════════════════════════════

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


def _rtqa_period(ex: Dict) -> str:
    import datetime as _dt
    raw = str(ex.get("question_date") or ex.get("date") or "")
    d   = re.sub(r"[^0-9]", "", raw)
    if not re.match(r"^\d{8}$", d):
        return "unknown"
    try:
        y, m, day = int(d[:4]), int(d[4:6]), int(d[6:8])
        iso = _dt.date(y, m, day).isocalendar()
        return f"{iso[0]:04d}-W{iso[1]:02d}"
    except ValueError:
        return "unknown"


def _rtqa_to_probe(ex: Dict, period: str) -> Dict:
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
        "source":     "realtimeqa",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RealtimeQA  — read raw/realtimeqa/train.parquet with pyarrow
# ══════════════════════════════════════════════════════════════════════════════

def _process_realtimeqa(n_probes: int, n_docs: int) -> None:
    parquet = RAW_ROOT / "realtimeqa" / "train.parquet"
    if not parquet.exists():
        print(f"[realtimeqa] {parquet} not found — skipping.")
        print("  Run: python Phase0/data/download_realtimeqa.py")
        return

    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        raise SystemExit(
            "[realtimeqa] pyarrow is required to read train.parquet.\n"
            "Install it with:  pip install pyarrow"
        )

    print(f"[realtimeqa] reading {parquet} …")
    table = pq.read_table(str(parquet))
    rows  = table.to_pydict()
    n_rows = table.num_rows
    print(f"  {n_rows:,} rows, columns: {table.column_names}")

    # Transpose column-dict → list of row-dicts
    col_names = list(rows.keys())
    by_period: Dict[str, Dict] = defaultdict(
        lambda: {"docs": [], "probes": [], "_idx": [0]}
    )

    for i in range(n_rows):
        ex  = {k: rows[k][i] for k in col_names}
        pid = _rtqa_period(ex)
        if pid == "unknown":
            continue

        bucket = by_period[pid]

        # Evidence / doc text
        evidence = ex.get("evidence") or ex.get("context") or ""
        if isinstance(evidence, list):
            evidence = " ".join(str(e) for e in evidence)
        ev_str = str(evidence).strip()

        if ev_str and len(ev_str) > 80 and len(bucket["docs"]) < n_docs:
            idx = bucket["_idx"][0]
            bucket["_idx"][0] += 1
            bucket["docs"].append(
                _make_stream_doc(ev_str, period=pid, source="realtimeqa",
                                 doc_idx=idx)
            )

        # Probe
        if len(bucket["probes"]) < n_probes:
            probe = _rtqa_to_probe(ex, pid)
            if probe["question"] and probe["answer"]:
                bucket["probes"].append(probe)

    timeline = sorted(by_period)
    out      = PROC_ROOT / "realtimeqa"
    counts: List[Tuple] = []

    for pid in timeline:
        d = by_period[pid]
        _write_jsonl(out / "stream" / f"{pid}.jsonl", d["docs"])
        _write_jsonl(out / "probes" / f"{pid}.jsonl", d["probes"])
        counts.append((pid, len(d["docs"]), len(d["probes"])))
        print(f"  · {pid}: {len(d['docs'])} docs, {len(d['probes'])} probes")

    _write_timeline(out, timeline, {
        "source":        "prajaktakini/realtime_qa",
        "period_scheme": "year-week",
        "probe_format":  "open_answer_mc",
        "counts":        counts,
    })
    print(f"[realtimeqa] done → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TemporalWiki  — re-derive probes from processed/temporalwiki/stream/
#
#     The original downloader streamed directly from HF and did not save raw
#     JSONL files to raw/temporalwiki/.  The processed stream/ files are
#     therefore the canonical local source for this dataset.
# ══════════════════════════════════════════════════════════════════════════════

def _process_temporalwiki(n_probes: int) -> None:
    stream_dir = PROC_ROOT / "temporalwiki" / "stream"
    if not stream_dir.exists() or not list(stream_dir.glob("*.jsonl")):
        print("[temporalwiki] processed/temporalwiki/stream/ is empty — skipping.")
        print("  Run: python Phase0/data/download_temporalwiki.py")
        return

    stream_files = sorted(stream_dir.glob("*.jsonl"))
    print(f"[temporalwiki] re-generating probes from {len(stream_files)} "
          f"stream file(s) …")

    timeline: List[str] = []
    counts:   List[Tuple] = []

    for sf in stream_files:
        pid    = sf.stem
        probes: List[Dict] = []

        for doc in _iter_jsonl(sf):
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            pr = _make_open_qa_probe(text, period=pid, source="temporalwiki")
            if pr is not None:
                probes.append(pr)
            if len(probes) >= n_probes:
                break

        out_path = PROC_ROOT / "temporalwiki" / "probes" / f"{pid}.jsonl"
        _write_jsonl(out_path, probes)
        timeline.append(pid)
        counts.append((pid, len(probes)))
        print(f"  · {pid}: {len(probes)} probes")

    out = PROC_ROOT / "temporalwiki"
    _write_timeline(out, timeline, {
        "source":        "seonghyeonye/TemporalWiki + wikimedia/wikipedia",
        "period_scheme": "snapshot",
        "probe_format":  "open_qa",
        "counts":        counts,
    })
    print(f"[temporalwiki] done → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  StreamingQA  — build from raw/cc_news/raw.jsonl (monthly slices)
#
#     StreamingQA (Liska et al., 2022) is based on CC-News.  We derive
#     temporal open-QA probes from the same source corpus using
#     make_open_qa_probe, giving us the same temporal structure (monthly
#     periods) without requiring the original dataset's annotations.
# ══════════════════════════════════════════════════════════════════════════════

def _cc_period(date_str: str) -> str:
    """Map a CC-News date string to a 'YYYY-MM' period id."""
    d = re.sub(r"[^0-9]", "", str(date_str or ""))
    if len(d) < 6:
        return "unknown"
    try:
        year, month = int(d[:4]), int(d[4:6])
    except ValueError:
        return "unknown"
    return f"{year:04d}-{month:02d}"


def _process_streaming_qa(n_docs: int, n_probes: int, n_periods: int) -> None:
    cc_raw = RAW_ROOT / "cc_news" / "raw.jsonl"
    if not cc_raw.exists():
        print(f"[streaming_qa] {cc_raw} not found — skipping.")
        print("  Run: python Phase0/data/download_cc_news.py")
        return

    # Apply a hard ceiling so we never OOM on the full 1.6 GB CC-News file.
    # The user can raise these with --max_* flags; the defaults are generous
    # for experiments (200 docs × 200 probes × 48 months ≈ 9.6K probes).
    eff_docs    = min(n_docs,    500)
    eff_probes  = min(n_probes,  500)
    eff_periods = min(n_periods, 48)

    print(f"[streaming_qa] building from {cc_raw} …")
    print(f"  caps: ≤{eff_periods} periods, "
          f"≤{eff_docs} docs/period, ≤{eff_probes} probes/period")

    # Use per-period file handles to avoid holding all data in RAM.
    out = PROC_ROOT / "streaming_qa"
    (out / "stream").mkdir(parents=True, exist_ok=True)
    (out / "probes").mkdir(parents=True, exist_ok=True)

    # Track state per period
    period_docs:    Dict[str, int] = defaultdict(int)   # count of docs written
    period_probes:  Dict[str, int] = defaultdict(int)   # count of probes written
    period_doc_idx: Dict[str, int] = defaultdict(int)
    period_handles_s: Dict[str, Any] = {}               # open stream file handles
    period_handles_p: Dict[str, Any] = {}               # open probe file handles
    period_order:   List[str] = []                       # insertion-order timeline

    def _open_period(pid: str) -> None:
        if pid not in period_handles_s:
            period_handles_s[pid] = open(out / "stream" / f"{pid}.jsonl", "w",
                                          encoding="utf-8")
            period_handles_p[pid] = open(out / "probes" / f"{pid}.jsonl", "w",
                                          encoding="utf-8")
            period_order.append(pid)

    def _close_all() -> None:
        for fh in list(period_handles_s.values()) + list(period_handles_p.values()):
            fh.close()

    try:
        for ex in _iter_jsonl(cc_raw):
            date_str = str(ex.get("date") or "")
            pid      = _cc_period(date_str)
            if pid == "unknown":
                continue

            # Skip if this period is already capped on both docs and probes
            docs_full   = period_docs[pid]   >= eff_docs
            probes_full = period_probes[pid] >= eff_probes
            if docs_full and probes_full:
                continue

            # Skip new periods once we have enough distinct periods
            if pid not in period_handles_s and len(period_order) >= eff_periods:
                continue

            text = (ex.get("text") or "").strip()
            if len(text) < 80:
                continue

            _open_period(pid)

            # Stream doc
            if not docs_full:
                idx = period_doc_idx[pid]
                period_doc_idx[pid] += 1
                doc = _make_stream_doc(
                    text, period=pid, source="streaming_qa", doc_idx=idx,
                    extra={"title": ex.get("title", ""), "url": ex.get("url", "")},
                )
                period_handles_s[pid].write(
                    json.dumps(doc, ensure_ascii=False) + "\n"
                )
                period_docs[pid] += 1

            # Open-QA probe
            if not probes_full:
                pr = _make_open_qa_probe(
                    text, period=pid, source="streaming_qa", date=date_str
                )
                if pr is not None:
                    period_handles_p[pid].write(
                        json.dumps(pr, ensure_ascii=False) + "\n"
                    )
                    period_probes[pid] += 1

            # Early exit: all desired periods are full on both docs and probes
            if len(period_order) >= eff_periods:
                if all(period_docs[p]   >= eff_docs and
                       period_probes[p] >= eff_probes
                       for p in period_order):
                    print("  [early exit] all periods filled.")
                    break
    finally:
        _close_all()

    timeline = sorted(period_order)
    counts: List[Tuple] = [
        (pid, period_docs[pid], period_probes[pid]) for pid in timeline
    ]
    for pid, nd, np_ in counts:
        print(f"  · {pid}: {nd} docs, {np_} probes")

    _write_timeline(out, timeline, {
        "source":             "raw/cc_news/raw.jsonl",
        "note":               ("Derived from CC-News; same source domain as "
                               "umilossegura/streamingqa"),
        "period_granularity": "month",
        "probe_format":       "open_qa",
        "counts":             counts,
    })
    print(f"[streaming_qa] done → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

ALL_DATASETS = ["realtimeqa", "temporalwiki", "streaming_qa"]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-process CAPSEL datasets from local raw files (no HF download).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--datasets", nargs="*", default=None, metavar="DS",
        help=f"Which datasets to process. Default: all "
             f"({', '.join(ALL_DATASETS)}).",
    )
    p.add_argument(
        "--max_docs_per_period", type=int, default=0, metavar="N",
        help="Max stream docs written per period (0 = no cap, but streaming_qa "
             "is internally capped at 500 to avoid OOM).",
    )
    p.add_argument(
        "--max_probes_per_period", type=int, default=0, metavar="N",
        help="Max probes written per period (0 = no cap).",
    )
    p.add_argument(
        "--max_periods", type=int, default=0, metavar="N",
        help="Max periods for streaming_qa (0 = no cap, internally capped at 48).",
    )
    args = p.parse_args()

    targets  = args.datasets or ALL_DATASETS
    n_docs   = _cap(args.max_docs_per_period)
    n_probes = _cap(args.max_probes_per_period)
    n_periods = _cap(args.max_periods)

    print(f"[preprocess_from_raw] targets : {targets}")
    print(f"[preprocess_from_raw] caps    : "
          f"docs={n_docs}, probes={n_probes}, periods={n_periods}")
    print()

    for ds in targets:
        if ds == "realtimeqa":
            _process_realtimeqa(n_probes, n_docs)
        elif ds == "temporalwiki":
            _process_temporalwiki(n_probes)
        elif ds == "streaming_qa":
            _process_streaming_qa(n_docs, n_probes, n_periods)
        else:
            print(f"[preprocess_from_raw] unknown dataset '{ds}' — skipping.")
        print()

    print("[preprocess_from_raw] all done.")


if __name__ == "__main__":
    main()
