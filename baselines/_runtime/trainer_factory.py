"""Seq2SeqTrainer factory + TrainingArguments builder.

Centralizes:
  * the ``tokenizer=`` vs ``processing_class=`` compatibility shim across
    transformers ≤4.46 / ≥4.47,
  * the bf16/fp16/cpu/Adafactor knobs read off ``INCAConfig``,
  * the standard ``DataCollatorForSeq2Seq`` (dynamic per-batch padding,
    label_pad_token_id=-100 to mask pad tokens in the loss).
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def _torch_version_str() -> str:
    return getattr(torch, "__version__", "?")


def _torch_version_tuple() -> Tuple[int, int]:
    """Return (major, minor) from torch.__version__ (ignores patch / +cuda tags)."""
    v = _torch_version_str().split("+", 1)[0]   # "2.5.1+cpu" → "2.5.1"
    parts = v.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return (0, 0)


def _mps_bf16_supported() -> bool:
    """Accelerate requires torch ≥ 2.6.0 for MPS + bf16 mixed precision."""
    maj, mn = _torch_version_tuple()
    return (maj, mn) >= (2, 6)


def _trainer_kwargs(model, tokenizer) -> Dict[str, Any]:
    """Compatibility shim for transformers ≤ 4.46 (``tokenizer=``) vs ≥ 4.47
    (renamed to ``processing_class=``).  Returns whichever keyword the
    installed version accepts."""
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    key = "processing_class" if "processing_class" in sig.parameters else "tokenizer"
    return {key: tokenizer}


def standard_trainer(
    model,
    args: Seq2SeqTrainingArguments,
    train_ds: TorchDataset,
    tokenizer,
) -> Seq2SeqTrainer:
    """Plain Seq2SeqTrainer with our standard collator.

    Used by every baseline that doesn't need a custom Trainer subclass
    (B1, B2, B4, B5, B6, B7).  B3 builds its own ``EWCTrainer`` and reuses the
    same collator + ``_trainer_kwargs`` shim directly.
    """
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, padding="longest", label_pad_token_id=-100,
        ),
        **_trainer_kwargs(model, tokenizer),
    )


def build_training_args(
    cfg, out_dir: Path, pid: str, seed: int, device: str,
) -> Seq2SeqTrainingArguments:
    """Build per-period ``Seq2SeqTrainingArguments`` from ``INCAConfig``.

    Notes on the precision / optimizer bundle (set in ``configs/base.yaml``):
      * ``precision=bf16``  — Trainer's ``bf16=True`` / ``bf16_full_eval=True``.
      * ``use_adafactor``   — Trainer's ``optim="adafactor"``.
      * CPU path forces ``use_cpu=True`` so Trainer doesn't auto-detect MPS.
      * ``save_strategy="no"`` because we use cross-period ``trainer.save_model``
        to persist a single HF-format best checkpoint (see TrainerRunner).
    """
    kw: Dict[str, Any] = dict(
        output_dir=str(out_dir / f"trainer_{pid}"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=getattr(cfg, "grad_accum_steps", 1),
        num_train_epochs=cfg.epochs_per_period,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=getattr(cfg, "warmup_ratio", 0.06),
        lr_scheduler_type="cosine",
        max_grad_norm=getattr(cfg, "max_grad_norm", 1.0),
        gradient_checkpointing=getattr(cfg, "gradient_checkpointing", False),
        logging_steps=getattr(cfg, "log_every_n_steps", 50),
        save_strategy="no",
        save_total_limit=1,
        report_to=[],
        disable_tqdm=True,    # TrainerLogCallback emits one line per logging_steps
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=seed,
    )
    if device == "cpu":
        kw["use_cpu"] = True

    prec = getattr(cfg, "precision", "bf16")
    # Safety: accelerate refuses bf16 on MPS unless torch ≥ 2.6.0.
    # Silently fall back to fp32 with a logged warning rather than crashing.
    if prec == "bf16" and device == "mps" and not _mps_bf16_supported():
        logging.getLogger("capsel.runner").warning(
            "torch %s on MPS does not support bf16 (need ≥ 2.6.0). "
            "Falling back to fp32. Upgrade with: pip install --upgrade 'torch>=2.6.0'",
            _torch_version_str(),
        )
        prec = "fp32"
    if device != "cpu":
        if prec == "bf16":
            kw["bf16"] = True
            kw["bf16_full_eval"] = True
        elif prec == "fp16":
            kw["fp16"] = True
            kw["fp16_full_eval"] = True

    # T5's training optimizer of choice: factored 2nd moment ≈ half the memory
    # of AdamW (m, v).  Quality on T5 is on par with AdamW.
    if getattr(cfg, "use_adafactor", False):
        kw["optim"] = "adafactor"

    return Seq2SeqTrainingArguments(**kw)
