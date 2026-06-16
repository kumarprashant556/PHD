"""Domain-sequential dataset loader  (data/domain_sequential.py)

Paper B three-domain curriculum following the LLaMA-Pro spirit.
Training order: P1_math → P2_code → P3_science

Domain      | HF source                               | n (default) | Framing
------------|------------------------------------------|-------------|---------------------------------------
P1_math     | lighteval/MATH  (config "all")           | 2 000       | "solve: " + problem → solution
P2_code     | bigcode/the-stack-smol  (Python)         | 2 000       | "complete: " + first_half → second_half
P3_science  | allenai/sciq                             | 2 000       | "answer: " + support + question → correct_answer

P1_math and P3_science use the natural problem/answer structure directly.
P2_code uses the standard completion split (no explicit Q/A structure in code files).
    period      = "P1_math" | "P2_code" | "P3_science"

Usage
-----
from data.domain_sequential import load_domain_sequential_periods
periods = load_domain_sequential_periods(n_per_period=2_000)
# {"P1_math": Dataset, "P2_code": Dataset, "P3_science": Dataset}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── HF import guard: prevent local datasets/ folder from shadowing the package ─
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_sp_backup = sys.path[:]
sys.path = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)]
from datasets import Dataset, concatenate_datasets, load_dataset  # noqa: E402
sys.path[:] = _sp_backup
del _sp_backup, _REPO_ROOT

from ._base import finalise, clean_text, subsample, drop_short, keep_columns, STANDARD_COLS

# ── Canonical period order ────────────────────────────────────────────────────

DEFAULT_PERIODS: List[str] = ["P1_math", "P2_code", "P3_science"]


# ── Per-domain raw loaders ────────────────────────────────────────────────────

def _load_math(n: int, seed: int) -> Dataset:
    """lighteval/MATH — natural problem → solution framing.

    The MATH dataset has ~12 500 training examples across algebra,
    counting & probability, geometry, number theory, and pre-calculus.
    Each example has: problem (str) · solution (str) · level · type · subject.
    We use the natural structure: input_text = "solve: " + problem,
    target_text = solution.  This is semantically correct for seq2seq and
    aligns with how GSM8K / MATH are evaluated in the LLaMA-Pro benchmarks.
    """
    print("  [domain_sequential] P1_math: loading lighteval/MATH …")
    try:
        ds = load_dataset("lighteval/MATH", "all", split="train", trust_remote_code=True)
    except Exception:
        # Fallback: try without config name (some HF versions omit it)
        ds = load_dataset("lighteval/MATH", split="train", trust_remote_code=True)

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        for prob, sol in zip(batch["problem"], batch["solution"]):
            inputs.append("solve: " + clean_text(str(prob or "")))
            targets.append(clean_text(str(sol or "")))
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  math: build input/target", remove_columns=ds.column_names)
    return ds


def _load_code(n: int, seed: int) -> Dataset:
    """bigcode/the-stack-smol (Python) — raw Python file completion.

    the-stack-smol is a deduplicated sample of GitHub code.  We take only
    Python files (the largest language slice, ~200k files even in the smol
    version) and treat each file as a freeform text document.  The completion
    split lands mid-function, encouraging generation of syntactically valid
    Python continuations.
    """
    print("  [domain_sequential] P2_code: loading bigcode/the-stack-smol Python …")
    try:
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: load full dataset and filter by lang column
        ds = load_dataset("bigcode/the-stack-smol", split="train", trust_remote_code=True)
        if "lang" in ds.column_names:
            ds = ds.filter(lambda x: (x.get("lang") or "").lower() == "python",
                           desc="  code: filter python")

    def _build_text(batch: dict) -> dict:
        texts = []
        content_col = "content" if "content" in batch else next(iter(batch))
        for content in batch[content_col]:
            texts.append(clean_text(str(content or "")))
        return {"text": texts}

    ds = ds.map(_build_text, batched=True, batch_size=512,
                desc="  code: build text", remove_columns=ds.column_names)
    # Drop very short files (< 30 words — filtered again in finalise() but
    # catching them early reduces map/subsample overhead on large splits).
    ds = ds.filter(lambda x: len(x["text"].split()) >= 30,
                   desc="  code: drop short files")
    return ds


def _load_science(n: int, seed: int) -> Dataset:
    """allenai/sciq — natural question → answer framing.

    SciQ has 13 679 science questions, each with a free-text support paragraph
    explaining the concept, the question, and the correct answer.
    We use the natural structure:
        input_text  = "answer: " + support_passage + " " + question
        target_text = correct_answer
    This avoids the arbitrary 50/50 split and makes the learning signal clean:
    the model reads the passage and question, then generates the answer.
    """
    print("  [domain_sequential] P3_science: loading allenai/sciq …")
    splits = []
    for split_name in ("train", "validation", "test"):
        try:
            splits.append(load_dataset("allenai/sciq", split=split_name))
        except Exception:
            pass
    if not splits:
        raise RuntimeError(
            "Could not load allenai/sciq from HuggingFace.  "
            "Check connectivity or pre-download with: "
            "datasets.load_dataset('allenai/sciq')"
        )
    ds = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        n_items = len(batch["question"])
        for support, question, answer in zip(
            batch.get("support", [""] * n_items),
            batch["question"],
            batch.get("correct_answer", [""] * n_items),
        ):
            ctx = clean_text(str(support or ""))
            q   = clean_text(str(question or ""))
            inp = "answer: " + " ".join(p for p in [ctx, q] if p)
            tgt = clean_text(str(answer or ""))
            inputs.append(inp)
            targets.append(tgt)
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  science: build input/target", remove_columns=ds.column_names)
    return ds


# ── Loader dispatch ───────────────────────────────────────────────────────────

_DOMAIN_LOADERS = {
    "P1_math":    _load_math,
    "P2_code":    _load_code,
    "P3_science": _load_science,
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_domain_sequential_periods(
    periods: Optional[List[str]] = None,
    n_per_period: int = 2_000,
    split_frac: float = 0.50,
    max_target_words: int = 200,
    seed: int = 42,
    num_proc: int = 4,
    **kwargs,
) -> Dict[str, Dataset]:
    """Load the Paper B three-domain curriculum and return period-keyed Datasets.

    Parameters
    ----------
    periods          : list of domain period IDs to return (default: all three in order)
    n_per_period     : maximum examples per domain after subsampling
    split_frac       : completion-split fraction (first_half → input_text)
    max_target_words : hard cap on decoder target length in words
    seed             : reproducibility seed (affects shuffle + subsample)
    num_proc         : parallel workers for Dataset.map

    Returns
    -------
    Dict[period_id, Dataset]   columns: input_text (str), target_text (str), period (str)

    Example
    -------
    >>> from data.domain_sequential import load_domain_sequential_periods
    >>> periods = load_domain_sequential_periods(n_per_period=500)
    >>> list(periods.keys())
    ['P1_math', 'P2_code', 'P3_science']
    >>> len(periods["P1_math"])   # ≤ 500 after filtering short docs
    ...
    """
    if periods is None:
        periods = DEFAULT_PERIODS

    period_set = set(periods)
    result: Dict[str, Dataset] = {}

    for period_id in DEFAULT_PERIODS:   # keep canonical training order
        if period_id not in period_set:
            continue

        loader = _DOMAIN_LOADERS[period_id]
        raw_ds = loader(n_per_period, seed)

        if "input_text" in raw_ds.column_names:
            # Natural problem/solution framing already applied (math, science).
            # Skip the completion split — just add period label, drop short
            # docs, subsample, and standardise columns.
            ds = raw_ds
            ds = ds.map(lambda _: {"period": period_id}, num_proc=num_proc,
                        desc=f"  {period_id}: add period label")
            ds = drop_short(ds, col="input_text",  min_len=20)
            ds = drop_short(ds, col="target_text", min_len=5)
            ds = subsample(ds, n_per_period, seed)
            final_ds = keep_columns(ds, STANDARD_COLS)
        else:
            # Completion-split framing (code — no explicit Q/A structure).
            final_ds = finalise(
                raw_ds,
                period=period_id,
                seed=seed,
                n=n_per_period,
                split_frac=split_frac,
                max_target_words=max_target_words,
                text_col="text",
                num_proc=num_proc,
            )

        result[period_id] = final_ds
        sample = final_ds[0]
        print(
            f"  {period_id}: {len(final_ds):,} examples  "
            f"(input ≈ {len(sample['input_text'].split()):,} words, "
            f"target ≈ {len(sample['target_text'].split()):,} words)"
        )

    return result
