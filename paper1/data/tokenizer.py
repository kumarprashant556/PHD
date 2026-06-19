"""FLAN-T5 tokenisation for seq2seq completion  (data/tokenizer.py)

Converts raw period Datasets (input_text, target_text, period) into
tokenized Datasets ready for a T5 training loop.

Key design choices
------------------
- Uses text_target= kwarg (replaces deprecated as_target_tokenizer())
- Padding on both encoder (input_ids / attention_mask) and decoder (labels)
- Decoder padding tokens replaced with -100 so CrossEntropyLoss ignores them
- Returns torch tensors when set_format is called

Usage
-----
from data.cc_news    import load_cc_news_periods
from data.tokenizer  import build_tokenized_periods, make_dataloader, make_replay_dataloader

raw     = load_cc_news_periods(n_per_period=20_000)
tok     = build_tokenized_periods(raw)

# Standard loader for one period
loader  = make_dataloader(tok["2018_H1"], batch_size=32)

# Replay-mixed loader: 75% stream + 25% replay buffer items
from data.tokenizer import make_replay_dataloader
mixed   = make_replay_dataloader(tok["2018_H1"], replay_items, batch_size=32)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize_batch(
    batch: Dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int = 256,
    max_target_length: int = 256,
) -> Dict[str, list]:
    """Tokenise a batch for FLAN-T5 encoder-decoder training.

    Returns variable-length sequences (no padding).  Padding is applied
    dynamically per batch by DataCollatorForSeq2Seq in the DataLoader,
    so each batch is padded only to its own longest sequence — not to
    the global max_length.  This reduces wasted compute significantly
    (MetaMathQA inputs average ~55 tokens vs the 256-token max).

    - Encoder : input_text  → input_ids, attention_mask
    - Decoder : target_text → labels  (DataCollatorForSeq2Seq replaces pad→-100)
    """
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding=False,          # dynamic padding at DataLoader time
    )
    # text_target= is the correct non-deprecated API (transformers ≥ 4.20)
    label_enc = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding=False,          # DataCollatorForSeq2Seq handles -100 replacement
    )
    model_inputs["labels"] = label_enc["input_ids"]
    return model_inputs


def build_tokenized_periods(
    period_datasets: Dict[str, Dataset],
    tokenizer_name: str = "google/flan-t5-base",
    max_input_length: int = 256,
    max_target_length: int = 256,
    num_proc: int = 4,
    batch_size: int = 1024,
    cache_dir: Optional[str] = None,
    tokenizer=None,
) -> Dict[str, Dataset]:
    """Tokenise all period datasets, with optional on-disk caching.

    Parameters
    ----------
    period_datasets  : output of any load_*_periods() function
    tokenizer_name   : HuggingFace tokeniser identifier
    max_input_length : encoder sequence length (tokens)
    max_target_length: decoder sequence length (tokens)
    num_proc         : parallel workers for Dataset.map
    batch_size       : map batch size (increase for faster tokenisation)
    cache_dir        : if set, save/load tokenised datasets here to skip
                       re-tokenising on subsequent runs (e.g. "cache/tokenized")

    Returns
    -------
    Dict[period_id, Dataset]
        Each Dataset has columns: input_ids, attention_mask, labels
        All formatted as torch tensors (set_format applied).
    """
    import hashlib
    from pathlib import Path as _Path
    from datasets import load_from_disk

    if tokenizer is None:
        print(f"Loading tokeniser: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenized: Dict[str, Dataset] = {}
    for period, ds in period_datasets.items():
        # ── Cache key: model name + lengths + period + dataset size ──────────
        if cache_dir is not None:
            key = hashlib.md5(
                f"{tokenizer_name}_{max_input_length}_{max_target_length}_{period}_{len(ds)}".encode()
            ).hexdigest()[:12]
            cache_path = _Path(cache_dir) / f"{period}_{key}"
            if cache_path.exists():
                tok_ds = load_from_disk(str(cache_path))
                tokenized[period] = tok_ds.with_format("python")
                print(f"  {period}: loaded from cache ({len(tok_ds):,} examples)")
                continue

        cols_to_remove = [c for c in ds.column_names
                          if c not in ("input_ids", "attention_mask", "labels")]
        tok_ds = ds.map(
            lambda batch: tokenize_batch(batch, tokenizer,
                                         max_input_length, max_target_length),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=cols_to_remove,
            desc=f"  tokenising {period}",
        )

        if cache_dir is not None:
            tok_ds.save_to_disk(str(cache_path))
            print(f"  {period}: saved to cache → {cache_path.name}")

        # Return items as plain Python lists (not numpy arrays).
        # HF Dataset's Arrow backend can return numpy arrays when iterated,
        # which triggers a slow `torch.tensor(list_of_numpy_arrays)` path in
        # DataCollatorForSeq2Seq and causes ~4 s/batch spikes.
        # with_format("python") forces pure Python types (list[int]) so the
        # collator takes the fast path.  We do NOT use set_format("torch")
        # because sequences have variable lengths and cannot be pre-stacked.
        tokenized[period] = tok_ds.with_format("python")
        print(f"  {period}: {len(tok_ds):,} tokenised examples")

    return tokenized


# ── DataLoaders ───────────────────────────────────────────────────────────────

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    collate_fn=None,
) -> DataLoader:
    """Standard DataLoader wrapping a tokenised Dataset.

    Pass collate_fn=DataCollatorForSeq2Seq(...) for dynamic padding.
    When None, PyTorch's default_collate is used (requires pre-padded tensors).

    num_workers=0 is intentional: MPS tensors cannot be shared across
    processes, so worker > 0 adds serialisation overhead and is slower
    on Apple Silicon. Prefetching is not needed when data is already in RAM.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,       # MPS does not use CUDA-style pinned memory
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class _ReplayMixDataset(IterableDataset):
    """Mixes stream items with replay buffer items in each batch.

    Each batch contains:
        - ceil(batch_size * (1 - replay_ratio)) stream items
        - floor(batch_size * replay_ratio) replay items

    Stream items are drawn sequentially (respects DataLoader epoch logic).
    Replay items are drawn uniformly at random from replay_items.
    """

    def __init__(
        self,
        stream_dataset: Dataset,
        replay_items: List[dict],
        batch_size: int = 32,
        replay_ratio: float = 0.25,
        seed: int = 42,
        collate_fn=None,
    ) -> None:
        self.stream     = stream_dataset
        self.replay     = replay_items
        self.bs         = batch_size
        self.ratio      = replay_ratio
        self.rng        = random.Random(seed)
        self.n_replay   = max(1, round(batch_size * replay_ratio))
        self.n_stream   = batch_size - self.n_replay
        self.collate_fn = collate_fn   # DataCollatorForSeq2Seq or None

    def __iter__(self):
        stream_iter = iter(self.stream)
        stream_buf: List[dict] = []

        while True:
            # Collect stream items
            while len(stream_buf) < self.n_stream:
                try:
                    stream_buf.append(next(stream_iter))
                except StopIteration:
                    if stream_buf:
                        break
                    return   # stream exhausted and buffer empty → done

            s_items = stream_buf[:self.n_stream]
            stream_buf = stream_buf[self.n_stream:]

            # Sample replay items
            if self.replay:
                r_items = self.rng.choices(self.replay, k=self.n_replay)
            else:
                r_items = []

            mixed = s_items + r_items
            self.rng.shuffle(mixed)

            # Collate into a single batch.
            # If a DataCollatorForSeq2Seq was provided, use it for dynamic
            # padding (each field padded to the longest item in the batch).
            # Otherwise fall back to torch.stack for pre-padded tensors.
            if self.collate_fn is not None:
                # Normalize every field to a plain Python list.
                # Stream items come from HF Dataset (Python lists or numpy arrays).
                # Replay buffer items were stored during training and may have
                # tensor fields.  DataCollatorForSeq2Seq's padding logic does
                #   label + [pad_id] * n
                # which fails on tensors but works on plain lists, so we must
                # convert before collation.
                def _to_list(v):
                    if isinstance(v, torch.Tensor):
                        return v.tolist()
                    if hasattr(v, "tolist"):          # numpy scalar / array
                        return v.tolist()
                    return v
                normalized = [{k: _to_list(v) for k, v in item.items()}
                              for item in mixed]
                yield self.collate_fn(normalized)
            else:
                batch: Dict[str, list] = {}
                for item in mixed:
                    for k, v in item.items():
                        batch.setdefault(k, []).append(v)
                for k in list(batch.keys()):
                    try:
                        batch[k] = torch.stack(batch[k])
                    except (TypeError, RuntimeError):
                        pass
                yield batch


def compute_percentile_length(
    period_datasets: Dict[str, Dataset],
    tokenizer,
    col: str = "target_text",
    percentile: float = 95.0,
    pad_to_multiple: int = 64,
    hard_cap: int = 512,
) -> int:
    """Scan *col* across all period datasets and return the P95 token length.

    The result is rounded up to the nearest multiple of *pad_to_multiple* and
    capped at *hard_cap*.  Pass ``col="input_text"`` to size the encoder instead.

    Typical values for domain_sequential with FLAN-T5:
      P1_math  (full solutions)  : P95 ≈ 250–350 tokens
      P2_code  (full programs)   : P95 ≈ 150–300 tokens
      P3_science (correct_answer): P95 ≈  10–20  tokens
      Combined P95               : driven by math / code → ≈ 192–320 tokens
    """
    import numpy as _np

    lengths = []
    for ds in period_datasets.values():
        texts = list(ds[col])
        # Tokenize one text at a time. Batch tokenization uses Rayon (Rust
        # thread pool) internally; a panic in any Rayon worker thread cannot
        # be caught by Python and kills the whole process with SIGSEGV.
        # Single-text calls stay on the Python thread and raise a normal
        # exception that we can swallow.
        for text in texts:
            try:
                enc = tokenizer(text, truncation=True, max_length=hard_cap,
                                padding=False, add_special_tokens=True)
                lengths.append(len(enc["input_ids"]))
            except Exception:
                pass

    p_len  = int(_np.percentile(lengths, percentile))
    rounded = ((p_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    return min(rounded, hard_cap)


def make_replay_dataloader(
    stream_dataset: Dataset,
    replay_items: List[dict],
    batch_size: int = 32,
    replay_ratio: float = 0.25,
    seed: int = 42,
    num_workers: int = 0,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader that mixes stream data with replay buffer samples.

    Parameters
    ----------
    stream_dataset : tokenised Dataset for the current period
    replay_items   : list of tokenised dicts from INCAReplayBuffer.all_items()
    batch_size     : total batch size (stream + replay combined)
    replay_ratio   : fraction of each batch drawn from replay buffer
    seed           : RNG seed for replay sampling
    num_workers    : DataLoader workers (keep 0 for IterableDataset)
    collate_fn     : DataCollatorForSeq2Seq (or None for pre-padded tensors)

    Returns
    -------
    DataLoader yielding mixed batches of shape (batch_size, seq_len)
    """
    mix_ds = _ReplayMixDataset(
        stream_dataset=stream_dataset,
        replay_items=replay_items,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        seed=seed,
        collate_fn=collate_fn,
    )
    return DataLoader(mix_ds, batch_size=None, num_workers=num_workers)
