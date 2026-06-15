"""Download TRACE proxy — five instruction-tuning tasks as CL periods.

TRACE (Wang et al, arXiv 2310.06762) is a continual-learning benchmark with
8 instruction-following tasks.  The original ``YuanQiQiqi/TRACE`` requires
authentication; we compose an equivalent multi-task stream from five
publicly accessible instruction datasets, one per domain:

  general        →  ``databricks/databricks-dolly-15k``
  math           →  ``gsm8k``  (main config)
  summarization  →  ``cnn_dailymail``  (3.0.0 config)
  coding         →  ``iamtarun/python_code_instructions_18k_alpaca``
  medical        →  ``medmcqa``

Raw output   → Phase0/data/raw/trace/<task>_train.parquet
                               /<task>_validation.parquet  (where available)
Processed    → Phase0/data/processed/trace/

Run::

    python Phase0/data/download_trace.py
    python Phase0/data/download_trace.py --max_docs_per_period 1000
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

from _utils import (
    cap,
    dataset_dir,
    make_completion_probe,
    make_doc,
    make_instruction_probe,
    raw_dir,
    require_hf_datasets,
    save_raw_parquet,
    write_jsonl,
    write_timeline,
)

DATASET_NAME = "trace"

# (period_id, hf_id, hf_config, train_split, eval_split)
TASKS: List[Tuple[str, str, Optional[str], str, str]] = [
    ("general",       "databricks/databricks-dolly-15k",             None,    "train", "train"),
    ("math",          "gsm8k",                                        "main",  "train", "test"),
    ("summarization", "cnn_dailymail",                                "3.0.0", "train", "validation"),
    ("coding",        "iamtarun/python_code_instructions_18k_alpaca", None,    "train", "train"),
    ("medical",       "medmcqa",                                      None,    "train", "validation"),
]


def _extract(ex: Dict, task: str) -> Tuple[str, str]:
    if task == "general":
        instr = (ex.get("instruction") or "").strip()
        ctx   = (ex.get("context") or "").strip()
        resp  = (ex.get("response") or "").strip()
        if ctx:
            instr = f"{instr}\n\nContext: {ctx}"
        return instr, resp
    if task == "math":
        return (ex.get("question") or "").strip(), (ex.get("answer") or "").strip()
    if task == "summarization":
        return ("Summarize the following article:\n\n"
                + (ex.get("article") or "").strip()), \
               (ex.get("highlights") or "").strip()
    if task == "coding":
        instr = (ex.get("instruction") or "").strip()
        inp   = (ex.get("input") or "").strip()
        out   = (ex.get("output") or "").strip()
        if inp:
            instr = f"{instr}\n\nInput: {inp}"
        return instr, out
    if task == "medical":
        q    = (ex.get("question") or "").strip()
        opts = [ex.get(f"op{c}", "") for c in ("a", "b", "c", "d")]
        cop  = ex.get("cop", 0)
        try:
            ans = opts[int(cop)]
        except (IndexError, TypeError, ValueError):
            ans = opts[0] if opts else ""
        exp  = (ex.get("exp") or "").strip()
        resp = f"{ans}. {exp}" if exp else str(ans)
        return q, resp.strip()
    return "", ""


def _build_mc4(question, answer, all_targets, rng, period) -> Optional[Dict]:
    if not question or not answer or not (1 <= len(answer) <= 120):
        return None
    distractors = [t for t in all_targets if t != answer]
    rng.shuffle(distractors)
    distractors = distractors[:3]
    if len(distractors) < 3:
        return None
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

    tasks = TASKS if n_periods >= len(TASKS) else TASKS[:n_periods]
    timeline = []
    counts = []

    for (pid, hf_id, config, train_split, eval_split) in tasks:
        print(f"[trace] loading task={pid!r} from {hf_id!r}…")
        load_kw: Dict[str, Any] = {}
        if config:
            load_kw["name"] = config
        # Load without cache_dir — HF Arrow cache stays in ~/.cache/huggingface
        ds = hf.load_dataset(hf_id, **load_kw)

        # Save original dataset splits as Parquet to raw/trace/
        # Prefix filenames with task name to avoid collisions
        try:
            import datasets as _hf  # type: ignore
            if isinstance(ds, _hf.DatasetDict):
                for sname, sds in ds.items():
                    dest = raw / f"{pid}_{sname}.parquet"
                    sds.to_parquet(str(dest))
                    mb = dest.stat().st_size / 1024 / 1024
                    print(f"  [raw] {pid}_{sname}.parquet  ({mb:.1f} MB)")
        except Exception as e:
            print(f"  [raw] Parquet export for {pid} failed: {e}")

        tr = train_split if train_split in ds else list(ds.keys())[0]
        ev = eval_split  if eval_split  in ds else tr

        docs: List[Dict] = []
        for ex in ds[tr]:
            instr, resp = _extract(ex, pid)
            if not instr or not resp or len(instr) + len(resp) < 30:
                continue
            text = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
            docs.append(make_doc(text, period=pid, source=DATASET_NAME,
                                 doc_idx=len(docs),
                                 extra={"format": "instruction"}))
            if len(docs) >= n_docs:
                break

        eval_items = list(ds[ev])
        rng.shuffle(eval_items)
        all_responses = []
        for ex in eval_items:
            _, resp = _extract(ex, pid)
            if resp and 1 <= len(resp) <= 120:
                all_responses.append(resp)

        probes: List[Dict] = []
        for ex in eval_items:
            if len(probes) >= n_probes:
                break
            instr, resp = _extract(ex, pid)
            if not instr or not resp:
                continue
            ip = make_instruction_probe(instr, resp, period=pid, source=DATASET_NAME)
            if ip:
                probes.append(ip)
            mc = _build_mc4(instr, resp, all_responses, rng, period=pid)
            if mc and len(probes) < n_probes:
                probes.append(mc)

        for d in docs[:max(1, len(docs) // 4)]:
            if len(probes) >= n_probes:
                break
            cp = make_completion_probe(d["text"], period=pid, source=DATASET_NAME)
            if cp:
                probes.append(cp)

        write_jsonl(out / "stream" / f"{pid}.jsonl", docs)
        write_jsonl(out / "probes" / f"{pid}.jsonl", probes)
        timeline.append(pid)
        counts.append((pid, len(docs), len(probes)))
        print(f"  · {pid}: {len(docs)} docs, {len(probes)} probes")

    write_timeline(out, timeline, {
        "source": "multi-task (dolly / gsm8k / cnn_dailymail / python_alpaca / medmcqa)",
        "period_scheme": "task",
        "probe_formats": ["mc4", "instruction", "completion"],
        "counts": counts,
    })
    print(f"[trace] done → {out}")


if __name__ == "__main__":
    main()
