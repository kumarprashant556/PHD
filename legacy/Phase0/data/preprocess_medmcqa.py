"""Preprocess raw MedMCQA Parquet splits into CAPSEL period JSONL.

Split roles
-----------
  train.parquet      (182k rows, labelled) → stream docs + training probes
  validation.parquet (4.2k  rows, labelled) → evaluation probes
  test.parquet       (6.1k  rows, no labels) → stream docs only

Period scheme (4 subject-group periods in curriculum order)
-----------------------------------------------------------
  basic_sciences   — anatomy, physiology, biochemistry, microbiology
  clinical_basics  — pharmacology, pathology
  clinical_applied — medicine, surgery, gynaecology & obstetrics, paediatrics
  specialties      — radiology, anaesthesia, forensic medicine, ophthalmology,
                     ent, psychiatry, dermatology, orthopaedics,
                     social & preventive medicine

Probe schema (unified CAPSEL format)
-------------------------------------
  {
    "question":   str,   # original MCQ question text
    "answer":     str,   # correct answer text (seq2seq generation target)
    "choices":    dict,  # {"A": ..., "B": ..., "C": ..., "D": ...}
    "answer_key": str,   # "A" | "B" | "C" | "D"  (original position, NOT rotated)
    "evidence":   str,   # explanation text from exp field (≤400 chars)
    "date":       str,   # "" — MedMCQA has no publication dates
    "period":     str,   # "basic_sciences" | "clinical_basics" | ...
    "source":     str,   # "medmcqa"
  }

Stream doc schema
------------------
  {
    "text":     str,    # "Question: …\n  A. …\n  B. …\nAnswer: X\nExplanation: …"
    "doc_id":   str,    # "medmcqa_<period>_<idx>"
    "period":   str,
    "source":   "medmcqa",
    "char_len": int,
    "subject":  str,
    "topic":    str,
  }

Dependencies: pip install pyarrow

Run::

    python Phase0/data/preprocess_medmcqa.py
    python Phase0/data/preprocess_medmcqa.py --force
    python Phase0/data/preprocess_medmcqa.py --max_docs_per_period 5000
    python Phase0/data/preprocess_medmcqa.py --max_probes_per_period 1000
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from _utils import (
    cap,
    dataset_dir,
    make_doc,
    raw_dir,
    write_jsonl,
    write_timeline,
)

DATASET_NAME = "medmcqa"
HF_ID        = "openlifescienceai/medmcqa"

# ── Subject → period mapping ──────────────────────────────────────────────────
SUBJECT_TO_PERIOD: Dict[str, str] = {
    "anatomy":                      "basic_sciences",
    "physiology":                   "basic_sciences",
    "biochemistry":                 "basic_sciences",
    "microbiology":                 "basic_sciences",
    "pharmacology":                 "clinical_basics",
    "pathology":                    "clinical_basics",
    "medicine":                     "clinical_applied",
    "surgery":                      "clinical_applied",
    "gynaecology & obstetrics":     "clinical_applied",
    "gynaecology":                  "clinical_applied",
    "obstetrics":                   "clinical_applied",
    "paediatrics":                  "clinical_applied",
    "pediatrics":                   "clinical_applied",
    "radiology":                    "specialties",
    "anaesthesia":                  "specialties",
    "anesthesia":                   "specialties",
    "forensic medicine":            "specialties",
    "ophthalmology":                "specialties",
    "ent":                          "specialties",
    "psychiatry":                   "specialties",
    "dermatology":                  "specialties",
    "orthopaedics":                 "specialties",
    "orthopedics":                  "specialties",
    "social & preventive medicine": "specialties",
    "social preventive medicine":   "specialties",
    "community medicine":           "specialties",
    "unknown":                      "clinical_basics",   # fallback
}

PERIOD_ORDER = [
    "basic_sciences",
    "clinical_basics",
    "clinical_applied",
    "specialties",
]

PERIOD_GROUPS = {
    "basic_sciences":  ["anatomy", "physiology", "biochemistry", "microbiology"],
    "clinical_basics": ["pharmacology", "pathology"],
    "clinical_applied":["medicine", "surgery", "gynaecology & obstetrics", "paediatrics"],
    "specialties":     ["radiology", "anaesthesia", "forensic medicine", "ophthalmology",
                        "ent", "psychiatry", "dermatology", "orthopaedics",
                        "social & preventive medicine"],
}

_KEYS = ["A", "B", "C", "D"]


def _subject_to_period(subject: str) -> str:
    s = (subject or "unknown").strip().lower()
    if s in SUBJECT_TO_PERIOD:
        return SUBJECT_TO_PERIOD[s]
    for key, period in SUBJECT_TO_PERIOD.items():
        if key in s or s in key:
            return period
    return "specialties"


# ── Row → probe (unified CAPSEL format) ──────────────────────────────────────

def _to_probe(ex: Dict, period: str) -> Optional[Dict]:
    """Convert a raw MedMCQA row to a unified CAPSEL probe dict.

    Preserves the original answer_key (A/B/C/D) without rotation.
    answer = text of the correct option (seq2seq generation target).
    Returns None if the row is missing a valid question or answer.
    """
    q   = (ex.get("question") or "").strip()
    opa = (ex.get("opa") or "").strip()
    opb = (ex.get("opb") or "").strip()
    opc = (ex.get("opc") or "").strip()
    opd = (ex.get("opd") or "").strip()
    cop = ex.get("cop")   # int 0-3: correct option index

    if not q or not opa:
        return None

    # Validate cop
    try:
        correct_idx = int(cop)
        if not (0 <= correct_idx <= 3):
            return None
    except (TypeError, ValueError):
        return None

    options = [opa, opb, opc, opd]
    choices = {_KEYS[i]: opt for i, opt in enumerate(options) if opt}
    if len(choices) < 2:
        return None

    answer_key  = _KEYS[correct_idx]
    answer_text = choices[answer_key]
    evidence    = (ex.get("exp") or "").strip()

    return {
        "question":   q,
        "answer":     answer_text,
        "choices":    choices,
        "answer_key": answer_key,
        "evidence":   evidence[:400] if evidence else "",
        "date":       "",
        "period":     period,
        "source":     DATASET_NAME,
    }


# ── Row → stream doc ──────────────────────────────────────────────────────────

def _to_stream_doc(ex: Dict, period: str, doc_idx: int) -> Optional[Dict]:
    """Convert a raw MedMCQA row to a stream doc.

    Format:
        Question: <question>
          A. <opa>
          B. <opb>
          C. <opc>
          D. <opd>
        Answer: <key>
        Explanation: <exp>

    Works for both labelled (train) and unlabelled (test) rows.
    For test rows cop may be absent — Answer line is omitted.
    """
    q   = (ex.get("question") or "").strip()
    opa = (ex.get("opa") or "").strip()
    opb = (ex.get("opb") or "").strip()
    opc = (ex.get("opc") or "").strip()
    opd = (ex.get("opd") or "").strip()
    exp = (ex.get("exp") or "").strip()
    cop = ex.get("cop")

    if not q or not opa:
        return None

    parts = [f"Question: {q}"]
    for label, opt in zip(_KEYS, [opa, opb, opc, opd]):
        if opt:
            parts.append(f"  {label}. {opt}")

    try:
        correct_idx = int(cop)
        if 0 <= correct_idx <= 3:
            parts.append(f"Answer: {_KEYS[correct_idx]}")
    except (TypeError, ValueError):
        pass  # test split: omit Answer line

    if exp:
        parts.append(f"Explanation: {exp}")

    text = "\n".join(parts)
    if len(text) < 60:
        return None

    return make_doc(
        text,
        period=period,
        source=DATASET_NAME,
        doc_idx=doc_idx,
        extra={
            "subject": (ex.get("subject_name") or "").strip(),
            "topic":   (ex.get("topic_name")   or "").strip(),
        },
    )


# ── Parquet reader (pyarrow) ──────────────────────────────────────────────────

def _read_parquet(path: Path) -> List[Dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        raise SystemExit(
            "[preprocess:medmcqa] pyarrow is required to read Parquet files.\n"
            "Install with:  pip install pyarrow"
        )
    table    = pq.read_table(str(path))
    col_dict = table.to_pydict()
    col_names = list(col_dict.keys())
    return [{k: col_dict[k][i] for k in col_names} for i in range(table.num_rows)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess raw MedMCQA Parquet splits → CAPSEL period JSONL."
    )
    p.add_argument("--max_periods", type=int, default=0,
                   help="Max subject-group periods (0 = all 4).")
    p.add_argument("--max_docs_per_period", type=int, default=0,
                   help="Max stream docs per period from train split (0 = all).")
    p.add_argument("--max_probes_per_period", type=int, default=0,
                   help="Max probes per period from train+validation (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Re-preprocess even if processed output already exists.")
    args = p.parse_args()

    n_periods = cap(args.max_periods)
    n_docs    = cap(args.max_docs_per_period)
    n_probes  = cap(args.max_probes_per_period)
    rng       = random.Random(args.seed)

    out = dataset_dir(DATASET_NAME)
    if not args.force and (out / "timeline.json").exists():
        print("[preprocess:medmcqa] already preprocessed — skipping  (--force to re-run)")
        return

    raw = raw_dir(DATASET_NAME)

    # ── Verify raw files ──────────────────────────────────────────────────────
    for split in ("train", "validation", "test"):
        p_ = raw / f"{split}.parquet"
        if not p_.exists():
            raise SystemExit(
                f"[preprocess:medmcqa] {p_} not found.\n"
                f"Run first: python Phase0/data/download_medmcqa.py"
            )

    # ── Load all splits ───────────────────────────────────────────────────────
    print("[preprocess:medmcqa] reading raw Parquet files…")
    train_rows = _read_parquet(raw / "train.parquet")
    val_rows   = _read_parquet(raw / "validation.parquet")
    test_rows  = _read_parquet(raw / "test.parquet")
    print(f"  train: {len(train_rows):,}  validation: {len(val_rows):,}"
          f"  test: {len(test_rows):,}")

    # ── Bucket rows by period ─────────────────────────────────────────────────
    # train → stream docs + training probes
    train_by_period: Dict[str, List[Dict]] = defaultdict(list)
    for ex in train_rows:
        pid = _subject_to_period(ex.get("subject_name", ""))
        train_by_period[pid].append(ex)

    # validation → eval probes
    val_by_period: Dict[str, List[Dict]] = defaultdict(list)
    for ex in val_rows:
        pid = _subject_to_period(ex.get("subject_name", ""))
        val_by_period[pid].append(ex)

    # test → extra stream docs (no labels, text is still useful)
    test_by_period: Dict[str, List[Dict]] = defaultdict(list)
    for ex in test_rows:
        pid = _subject_to_period(ex.get("subject_name", ""))
        test_by_period[pid].append(ex)

    # ── Process each period ───────────────────────────────────────────────────
    active_periods = PERIOD_ORDER[:n_periods] if n_periods < len(PERIOD_ORDER) \
                     else PERIOD_ORDER
    timeline: List[str] = []
    counts:   List[Tuple] = []

    for pid in active_periods:
        t_exs    = train_by_period.get(pid, [])
        v_exs    = val_by_period.get(pid, [])
        te_exs   = test_by_period.get(pid, [])

        rng.shuffle(t_exs)
        rng.shuffle(v_exs)

        # ── Stream docs: train + test (both have rich text) ──────────────────
        docs: List[Dict] = []
        doc_idx = 0
        for ex in t_exs:
            if len(docs) >= n_docs:
                break
            d = _to_stream_doc(ex, pid, doc_idx)
            if d:
                docs.append(d)
                doc_idx += 1

        # Append test docs after train docs (no labels but text is useful)
        for ex in te_exs:
            if len(docs) >= n_docs:
                break
            d = _to_stream_doc(ex, pid, doc_idx)
            if d:
                docs.append(d)
                doc_idx += 1

        # ── Probes: validation first (clean eval set), then train ─────────────
        # validation = held-out evaluation probes
        # train = additional training probes
        probes: List[Dict] = []

        for ex in v_exs:
            if len(probes) >= n_probes:
                break
            pr = _to_probe(ex, pid)
            if pr:
                probes.append(pr)

        for ex in t_exs:
            if len(probes) >= n_probes:
                break
            pr = _to_probe(ex, pid)
            if pr:
                probes.append(pr)

        # Shuffle so val+train probes are interleaved for training
        rng.shuffle(probes)

        write_jsonl(out / "stream" / f"{pid}.jsonl", docs)
        write_jsonl(out / "probes" / f"{pid}.jsonl", probes)
        timeline.append(pid)
        counts.append((pid, len(docs), len(probes)))
        print(f"  · {pid}: {len(docs):,} stream docs, {len(probes):,} probes")

    write_timeline(out, timeline, {
        "source":         HF_ID,
        "period_scheme":  "subject_group",
        "probe_format":   "open_answer_mc",
        "splits_used": {
            "stream": ["train", "test"],
            "probes": ["validation", "train"],
        },
        "period_groups":  PERIOD_GROUPS,
        "counts":         counts,
    })
    print(f"[preprocess:medmcqa] done → {out}")


if __name__ == "__main__":
    main()
