"""TRACE period loader  (data/trace.py)

Multi-domain continual learning benchmark (secondary evaluation).

Source     : 5 public instruction datasets, one per domain
Coverage   : general, math, summarisation, coding, medical
Period key : domain name  e.g. "general", "math"

Framing strategy
-----------------
TRACE is an instruction-following benchmark. We adapt each domain to
seq2seq completion:
  - instruction/question  →  encoder input ("answer: <instruction>")
  - answer/solution       →  decoder target

For summarisation (CNN/DM): encoder = "summarise: <article>", target = highlights
For coding: encoder = "complete code: <prompt>", target = <solution>

Original TRACE (Wang et al, 2023, arXiv 2310.06762): 8 tasks.
This loader covers 5 publicly accessible domains:
    general        databricks/databricks-dolly-15k
    math           gsm8k  (main)
    summarisation  cnn_dailymail  (3.0.0)
    coding         iamtarun/python_code_instructions_18k_alpaca
    medical        openlifescienceai/medmcqa

Usage
-----
from data.trace import load_trace_periods

periods = load_trace_periods(n_per_period=5_000)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from datasets import Dataset, load_dataset

from ._base import subsample, drop_short, keep_columns, clean_text, STANDARD_COLS

# ── Per-domain configs ────────────────────────────────────────────────────────

_DOMAINS: Dict[str, dict] = {
    "general": {
        "hf_id":   "databricks/databricks-dolly-15k",
        "split":   "train",
        "input_template":  "answer: {instruction} {context}",
        "target_col": "response",
        "prefix": "answer: ",
    },
    "math": {
        "hf_id":   "gsm8k",
        "config":  "main",
        "split":   "train",
        "input_template":  "solve: {question}",
        "target_col": "answer",
        "prefix": "solve: ",
    },
    "summarisation": {
        "hf_id":   "cnn_dailymail",
        "config":  "3.0.0",
        "split":   "train",
        "input_template":  "summarise: {article}",
        "target_col": "highlights",
        "prefix": "summarise: ",
        "truncate_input_words": 200,   # CNN/DM articles are long
    },
    "coding": {
        "hf_id":   "iamtarun/python_code_instructions_18k_alpaca",
        "split":   "train",
        "input_template":  "complete code: {instruction} {input}",
        "target_col": "output",
        "prefix": "complete code: ",
    },
    "medical": {
        "hf_id":   "openlifescienceai/medmcqa",
        "split":   "train",
        "input_template":  "answer: {question}",
        "target_col": None,          # build from cop (correct option) + options
        "prefix": "answer: ",
    },
}

DEFAULT_DOMAINS: List[str] = list(_DOMAINS.keys())

# ── Domain-specific row converters ────────────────────────────────────────────

def _format_row(row: dict, cfg: dict) -> Dict[str, str]:
    """Convert a raw HF row to (input_text, target_text) for one domain."""
    prefix = cfg.get("prefix", "answer: ")
    trunc  = cfg.get("truncate_input_words", None)

    # Build input text
    template = cfg["input_template"]
    try:
        # Fill template fields; missing keys become empty string
        filled = template.format_map({k: clean_text(str(v or ""))
                                       for k, v in row.items()})
    except Exception:
        filled = prefix + clean_text(str(row.get("instruction",
                                        row.get("question", ""))))

    if trunc:
        words = filled.split()
        filled = " ".join(words[:trunc])

    # Build target text
    target_col = cfg.get("target_col")
    if target_col is None:
        # Medical: reconstruct correct answer from option fields
        cop   = row.get("cop", 0) or 0          # 0-indexed correct option
        opts  = [row.get(f"op{i}", "") or "" for i in range(1, 5)]
        try:
            target = clean_text(opts[int(cop)])
        except (IndexError, ValueError):
            target = ""
    else:
        target = clean_text(str(row.get(target_col, "") or ""))

    return {"input_text": filled, "target_text": target}


# ── Public API ────────────────────────────────────────────────────────────────

def load_trace_periods(
    domains: Optional[List[str]] = None,
    n_per_period: int = 5_000,
    seed: int = 42,
    num_proc: int = 4,
    cache_dir: Optional[str] = None,
) -> Dict[str, Dataset]:
    """Load TRACE-equivalent domains as a period-keyed Dataset dict.

    Each domain becomes one period in the continual learning stream.
    Order follows DEFAULT_DOMAINS (general → math → summarisation →
    coding → medical), reflecting increasing task specificity.

    Parameters
    ----------
    domains      : subset of DEFAULT_DOMAINS to include
    n_per_period : max examples per domain
    seed         : random seed for subsample
    num_proc     : parallel workers for map
    cache_dir    : HuggingFace cache directory

    Returns
    -------
    Dict[domain_name, Dataset]  — columns: input_text, target_text, period
    """
    if domains is None:
        domains = DEFAULT_DOMAINS

    result: Dict[str, Dataset] = {}
    for domain in domains:
        cfg = _DOMAINS[domain]
        print(f"  Loading TRACE/{domain} ({cfg['hf_id']}) …")
        try:
            hf_kwargs = dict(split=cfg["split"], cache_dir=cache_dir)
            if "config" in cfg:
                hf_kwargs["name"] = cfg["config"]
            ds = load_dataset(cfg["hf_id"], **hf_kwargs)
        except Exception as e:
            print(f"  WARNING: could not load {domain}: {e} — skipping")
            continue

        # Convert each row to (input_text, target_text)
        def convert(batch, cfg=cfg):
            inputs, targets = [], []
            for i in range(len(batch[list(batch.keys())[0]])):
                row = {k: batch[k][i] for k in batch}
                out = _format_row(row, cfg)
                inputs.append(out["input_text"])
                targets.append(out["target_text"])
            return {"input_text": inputs, "target_text": targets,
                    "period": [domain] * len(inputs)}

        ds = ds.map(convert, batched=True, batch_size=512, num_proc=num_proc,
                    desc=f"    converting {domain}")

        ds = drop_short(ds, col="input_text",  min_len=10)
        ds = drop_short(ds, col="target_text", min_len=2)
        ds = subsample(ds, n_per_period, seed)
        ds = keep_columns(ds, STANDARD_COLS)
        result[domain] = ds
        print(f"  {domain}: {len(ds):,} examples ready")

    return result
