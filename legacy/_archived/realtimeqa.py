"""RealtimeQA period loader  (data/realtimeqa.py)

Secondary evaluation dataset (Phase 0 Track A).

Source     : prajaktakini/realtime_qa  (HuggingFace)
Coverage   : weekly MC + open-answer QA, questions about recent news
Period key : "YYYY-Www"  (ISO year + week)  e.g. "2022-W14"
F1 ceiling : ~30–35% with FLAN-T5-base (used as evaluation only, not pretraining)

Framing strategy
-----------------
RealtimeQA is a QA dataset, not a completion corpus. We adapt it to
seq2seq completion mode by treating the question as encoder input and
the correct answer choice as decoder target:

    input_text  = "answer: <question>"
    target_text = "<correct answer text>"

This keeps it in T5's text-to-text paradigm while measuring temporal
knowledge update.

Usage
-----
from data.realtimeqa import load_realtimeqa_periods

periods = load_realtimeqa_periods(n_per_period=2_000)
# smaller n — RealtimeQA has ~200–400 questions per week
"""

from __future__ import annotations

from typing import Dict, List, Optional

from datasets import Dataset, load_dataset

from ._base import subsample, drop_short, keep_columns, clean_text, STANDARD_COLS

# ── Framing ───────────────────────────────────────────────────────────────────

QA_PREFIX = "answer: "


def _to_qa_completion(example: dict) -> dict:
    """Convert a RealtimeQA example to seq2seq QA framing."""
    question = clean_text(example.get("question_sentence", "") or "")
    choices  = example.get("choices", [])
    answer_i = example.get("answer", None)

    # answer field can be int index or string
    answer_text = ""
    if answer_i is not None and choices:
        try:
            answer_text = clean_text(choices[int(answer_i)])
        except (IndexError, ValueError, TypeError):
            answer_text = ""

    return {
        "input_text":  QA_PREFIX + question if question else "",
        "target_text": answer_text,
    }


def _extract_week(example: dict) -> dict:
    """Extract ISO week key 'YYYY-Www' from the date field."""
    try:
        from datetime import datetime, date
        d_str = example.get("question_date", "") or ""
        d = datetime.strptime(d_str[:10], "%Y-%m-%d").date()
        y, w, _ = d.isocalendar()
        return {"period": f"{y}-W{w:02d}"}
    except Exception:
        return {"period": "unknown"}


# ── Period grouping ───────────────────────────────────────────────────────────

def _group_by_period(ds: Dataset) -> Dict[str, Dataset]:
    """Split dataset into per-week sub-datasets."""
    periods: Dict[str, list] = {}
    for row in ds:
        p = row.get("period", "unknown")
        periods.setdefault(p, []).append(row)
    return {p: Dataset.from_list(rows)
            for p, rows in sorted(periods.items())
            if p != "unknown"}


# ── Public API ────────────────────────────────────────────────────────────────

def load_realtimeqa_periods(
    n_per_period: int = 2_000,
    seed: int = 42,
    hf_id: str = "prajaktakini/realtime_qa",
    num_proc: int = 4,
    max_periods: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Dataset]:
    """Load RealtimeQA and group by ISO week.

    Parameters
    ----------
    n_per_period : max QA examples per week (RealtimeQA has ~200–400/week)
    seed         : random seed for subsample
    hf_id        : HuggingFace dataset ID
    max_periods  : if set, only return the most recent *max_periods* weeks
    cache_dir    : HuggingFace cache directory

    Returns
    -------
    Dict[week_id, Dataset]  — columns: input_text, target_text, period
    """
    print(f"Loading {hf_id} …")
    try:
        ds = load_dataset(hf_id, split="validation", cache_dir=cache_dir)
    except Exception:
        ds = load_dataset(hf_id, split="train", cache_dir=cache_dir)

    print(f"  total QA examples: {len(ds):,}")

    # Convert to seq2seq framing + extract week label
    ds = ds.map(_to_qa_completion, num_proc=num_proc, desc="  QA framing")
    ds = ds.map(_extract_week,     num_proc=num_proc, desc="  week labels")

    # Group by week
    period_map = _group_by_period(ds)
    if max_periods:
        # Keep most recent weeks
        keys = sorted(period_map)[-max_periods:]
        period_map = {k: period_map[k] for k in keys}

    result: Dict[str, Dataset] = {}
    for period, subset in period_map.items():
        subset = drop_short(subset, col="input_text",  min_len=len(QA_PREFIX) + 5)
        subset = drop_short(subset, col="target_text", min_len=2)
        subset = subsample(subset, n_per_period, seed)
        subset = keep_columns(subset, STANDARD_COLS)
        result[period] = subset

    print(f"  {len(result)} weekly periods loaded")
    if result:
        sizes = [len(v) for v in result.values()]
        print(f"  examples/period: min={min(sizes)}, max={max(sizes)}, "
              f"mean={sum(sizes)//len(sizes)}")
    return result
