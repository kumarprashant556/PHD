"""Download CKL proxy — two knowledge-probe periods from open QA datasets.

CKL (Jang et al, ICLR 2022) tests knowledge retention (InvariantLAMA),
update (UpdatedLAMA), and acquisition (NewLAMA).  The original
``joeyoonjeong/CKL`` requires authentication; we build an equivalent
two-period layout from fully public sources:

  period_A  →  ``web_questions``  (Berant et al, 2013 — older entity facts)
  period_B  →  ``trivia_qa``      (Joshi et al, 2017 — broader fact coverage)

Raw output   → Phase0/data/raw/ckl/web_questions_train.parquet
                               /trivia_qa_train.parquet
Processed    → Phase0/data/processed/ckl/

Run::

    python Phase0/data/download_ckl.py
    python Phase0/data/download_ckl.py --max_docs_per_period 500 --probes_per_period 200
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import random
from typing import Dict, List, Optional

from _utils import (
    cap,
    dataset_dir,
    make_completion_probe,
    make_doc,
    raw_dir,
    require_hf_datasets,
    save_raw_parquet,
    write_jsonl,
    write_timeline,
)

DATASET_NAME = "ckl"


def _make_mc4(question, answer, all_answers, rng, period) -> Optional[Dict]:
    if not question or not answer:
        return None
    distractors = [a for a in all_answers if a.lower() != answer.lower()]
    rng.shuffle(distractors)
    distractors = distractors[:3]
    while len(distractors) < 3:
        distractors.append("n/a")
    keys = ["A", "B", "C", "D"]
    return {
        "format": "mc4",
        "question": question,
        "evidence": "",
        "choices": {keys[i]: c for i, c in enumerate([answer] + distractors)},
        "answer_key": "A",
        "period": period,
        "source": DATASET_NAME,
    }


def _load_web_questions(hf, raw, period, n_docs, n_probes, rng):
    print(f"[ckl] loading web_questions…")
    # Load without cache_dir — HF Arrow cache stays in ~/.cache/huggingface
    ds = hf.load_dataset("web_questions")

    # Save original dataset as Parquet to raw/ckl/
    dest = raw / "web_questions_train.parquet"
    split = "train" if "train" in ds else list(ds.keys())[0]
    ds[split].to_parquet(str(dest))
    mb = dest.stat().st_size / 1024 / 1024
    print(f"  [raw] web_questions_train.parquet  ({mb:.1f} MB) → {dest}")

    rows = list(ds[split])
    all_answers = [r["answers"][0] for r in rows
                   if r.get("answers") and r["answers"][0]]

    docs: List[Dict] = []
    probes: List[Dict] = []

    for ex in rows:
        q = (ex.get("question") or "").strip()
        ans_list = ex.get("answers") or []
        answer = ans_list[0].strip() if ans_list else ""

        if len(docs) < n_docs and q and answer:
            text = f"Q: {q}\nA: {answer}"
            docs.append(make_doc(text, period=period, source=DATASET_NAME,
                                 doc_idx=len(docs)))

        if len(probes) < n_probes:
            mc = _make_mc4(q, answer, all_answers, rng, period)
            if mc:
                probes.append(mc)

    return docs, probes


def _load_trivia_qa(hf, raw, period, n_docs, n_probes, rng):
    print(f"[ckl] loading trivia_qa…")
    # Load without cache_dir — HF Arrow cache stays in ~/.cache/huggingface
    ds = hf.load_dataset("trivia_qa", name="rc", streaming=True)
    split = "train" if "train" in ds else list(ds.keys())[0]

    rows: List[Dict] = []
    for ex in ds[split]:
        rows.append(ex)
        if len(rows) >= max(n_probes * 4, 2000):
            break

    # Save sampled rows as Parquet
    try:
        import pandas as pd  # type: ignore
        dest = raw / "trivia_qa_train.parquet"
        pd.DataFrame(rows).to_parquet(str(dest), index=False)
        mb = dest.stat().st_size / 1024 / 1024
        print(f"  [raw] trivia_qa_train.parquet  ({mb:.1f} MB) → {dest}")
    except Exception as e:
        print(f"  [raw] trivia_qa Parquet export skipped: {e}")

    all_answers = []
    for r in rows:
        ans = r.get("answer") or {}
        val = (ans.get("value") or "").strip() if isinstance(ans, dict) else ""
        if val:
            all_answers.append(val)

    docs: List[Dict] = []
    probes: List[Dict] = []

    for ex in rows:
        q = (ex.get("question") or "").strip()
        ans = ex.get("answer") or {}
        answer = (ans.get("value") or "").strip() if isinstance(ans, dict) else ""

        if len(docs) < n_docs:
            sr = ex.get("search_results") or {}
            if isinstance(sr, dict):
                passages = sr.get("search_context") or sr.get("description") or []
                if isinstance(passages, list):
                    for passage in passages:
                        if isinstance(passage, str) and len(passage) > 100:
                            docs.append(make_doc(passage, period=period,
                                                 source=DATASET_NAME,
                                                 doc_idx=len(docs)))
                            if len(docs) >= n_docs:
                                break

        if len(probes) < n_probes and q and answer:
            mc = _make_mc4(q, answer, all_answers, rng, period)
            if mc:
                probes.append(mc)

    if not docs:
        for ex in rows[:n_docs]:
            q = (ex.get("question") or "").strip()
            ans = ex.get("answer") or {}
            answer = (ans.get("value") or "").strip() if isinstance(ans, dict) else ""
            if q and answer:
                text = f"Q: {q}\nA: {answer}"
                docs.append(make_doc(text, period=period, source=DATASET_NAME,
                                     doc_idx=len(docs)))

    return docs, probes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_periods", type=int, default=0)
    p.add_argument("--max_docs_per_period", type=int, default=0)
    p.add_argument("--probes_per_period", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    n_periods = cap(args.max_periods)
    n_docs    = cap(args.max_docs_per_period)
    n_probes  = cap(args.probes_per_period)
    rng = random.Random(args.seed)

    hf = require_hf_datasets()
    raw = raw_dir(DATASET_NAME)
    out = dataset_dir(DATASET_NAME)

    period_loaders = [
        ("period_A", _load_web_questions),
        ("period_B", _load_trivia_qa),
    ]
    if n_periods < len(period_loaders):
        period_loaders = period_loaders[:n_periods]

    timeline = []
    counts = []

    for pid, loader in period_loaders:
        docs, probes = loader(hf, raw, pid, n_docs, n_probes, rng)

        comp_budget = max(0, n_probes - len(probes))
        for d in docs[:comp_budget]:
            cp = make_completion_probe(d["text"], period=pid, source=DATASET_NAME)
            if cp:
                probes.append(cp)

        write_jsonl(out / "stream" / f"{pid}.jsonl", docs)
        write_jsonl(out / "probes" / f"{pid}.jsonl", probes)
        timeline.append(pid)
        counts.append((pid, len(docs), len(probes)))
        print(f"  · {pid}: {len(docs)} docs, {len(probes)} probes")

    write_timeline(out, timeline, {
        "source": "web_questions (period_A) + trivia_qa (period_B)",
        "period_scheme": "snapshot",
        "probe_formats": ["mc4", "completion"],
        "counts": counts,
    })
    print(f"[ckl] done → {out}")


if __name__ == "__main__":
    main()
