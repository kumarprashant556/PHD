"""capsel.data — unified dataset registry.

All loaders return  Dict[period_id, datasets.Dataset]
with columns:  input_text (str) · target_text (str) · period (str)

Only seq2seq text-completion datasets are registered.
QA / MCQ datasets (ckl, realtimeqa, medmcqa, trace) are excluded.

Quick start
-----------
from data import load_periods, tokenize

# Load CC-News (Phase 1 default)
raw = load_periods("cc_news", n_per_period=20_000)

# Tokenise for FLAN-T5
tok = tokenize(raw)

# DataLoader for one period
from data import make_dataloader
loader = make_dataloader(tok["2018_H1"], batch_size=32)

Dataset registry (text-completion only)
----------------------------------------
"cc_news"             local local_data/cc_news/             Phase 1 primary  (4 half-year periods)
"streaming_qa"        local local_data/streaming_qa/        CC-News corpus   (26 monthly periods)
"temporalwiki"        local local_data/temporalwiki/        Wikipedia snaps  (2 periods)
"tic_lm"              local local_data/tic_lm/              C4 daily slices  (9 periods)
"redpajama"           RedPajama-Data-V2 (HF streaming)    E-ROUTE benchmark
"domain_sequential"   HF: MATH + the-stack-smol + SciQ    Paper B primary  (3 domains)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from datasets import Dataset
except Exception:  # pragma: no cover - allows lightweight preprocessing without HF deps
    Dataset = Any  # type: ignore

# ── Loader registry ───────────────────────────────────────────────────────────

_LOADERS = {
    # ── Text-completion datasets (seq2seq training) ───────────────────────────
    "cc_news":            ("data.cc_news",            "load_cc_news_periods"),
    "streaming_qa":       ("data.streaming_qa",       "load_streaming_qa_periods"),
    "temporalwiki":       ("data.temporalwiki",       "load_temporalwiki_periods"),
    "tic_lm":             ("data.tic_lm",             "load_tic_lm_periods"),
    "redpajama":          ("data.redpajama",          "load_redpajama_periods"),
    # ── Paper B: domain-sequential curriculum (math → code → science) ────────
    "domain_sequential":  ("data.domain_sequential",  "load_domain_sequential_periods"),
    # ── QA / MCQ datasets excluded from training ─────────────────────────────
    # "ckl":        excluded — TriviaQA/WebQuestions MCQ format
    # "realtimeqa": excluded — weekly news QA/MCQ format
    # "medmcqa":    excluded — medical MCQ format
    # "trace":      excluded — instruction-following format
}


def load_periods(
    dataset: str,
    **kwargs,
) -> Dict[str, Dataset]:
    """Load a dataset by name and return period-keyed Datasets.

    Parameters
    ----------
    dataset : one of cc_news | streaming_qa | temporalwiki | tic_lm | redpajama
    **kwargs : forwarded to the dataset-specific loader

    Returns
    -------
    Dict[period_id, Dataset]  — columns: input_text, target_text, period
    """
    if dataset not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {sorted(_LOADERS)}"
        )
    module_name, fn_name = _LOADERS[dataset]
    import importlib
    mod = importlib.import_module(module_name)
    loader_fn = getattr(mod, fn_name)
    return loader_fn(**kwargs)


# ── Tokenisation helpers (re-exported) ───────────────────────────────────────

def tokenize(
    period_datasets: Dict[str, Dataset],
    tokenizer_name: str = "google/flan-t5-base",
    max_input_length: int = 256,
    max_target_length: int = 256,
    num_proc: int = 4,
) -> Dict[str, Dataset]:
    """Tokenise period datasets for FLAN-T5.  Re-exports tokenizer.build_tokenized_periods."""
    from data.tokenizer import build_tokenized_periods
    return build_tokenized_periods(
        period_datasets,
        tokenizer_name=tokenizer_name,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        num_proc=num_proc,
    )


def make_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True,
                    num_workers: int = 0):
    """Convenience re-export of tokenizer.make_dataloader."""
    from data.tokenizer import make_dataloader as _dl
    return _dl(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def make_replay_dataloader(stream_dataset: Dataset, replay_items: list,
                            batch_size: int = 32, replay_ratio: float = 0.25,
                            seed: int = 42, num_workers: int = 0):
    """Convenience re-export of tokenizer.make_replay_dataloader."""
    from data.tokenizer import make_replay_dataloader as _rdl
    return _rdl(stream_dataset, replay_items, batch_size=batch_size,
                replay_ratio=replay_ratio, seed=seed, num_workers=num_workers)


__all__ = [
    "load_periods",
    "tokenize",
    "make_dataloader",
    "make_replay_dataloader",
]
