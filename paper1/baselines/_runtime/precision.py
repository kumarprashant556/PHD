"""Precision helpers — pick training / autocast dtype by config + device."""
from __future__ import annotations

from typing import Optional

import torch


def model_dtype(device: str, cfg=None) -> torch.dtype:
    """Pick the *model* training dtype based on ``cfg.precision`` (default bf16).

    Precision choices:
      * "bf16"  — bfloat16.  Numerically stable on T5; ~2× faster matmul on
                  M2+ Metal and Ampere+ CUDA.  Half the memory of fp32.
                  **CPU does not support bf16 reliably across all ops**, so we
                  fall back to fp32 on cpu.
      * "fp16"  — float16.  Fastest on older CUDA, but T5 frequently produces
                  NaN in attention.  Avoid unless you know what you're doing.
      * "fp32"  — full precision.  Slowest, largest memory.  Use for debugging.
    """
    prec = getattr(cfg, "precision", "bf16") if cfg is not None else "bf16"
    if prec == "fp32" or device == "cpu":
        return torch.float32
    if prec == "fp16":
        return torch.float16
    return torch.bfloat16


def autocast_dtype(cfg, device: str) -> Optional[torch.dtype]:
    """Return the dtype to use with ``torch.autocast`` for eval, or None to skip.

    Mirrors training precision so that eval generation runs at the same speed
    as training.  CPU has no autocast benefit, so we return None there.
    """
    if device == "cpu":
        return None
    prec = getattr(cfg, "precision", "bf16") if cfg is not None else "bf16"
    if prec == "bf16":
        return torch.bfloat16
    if prec == "fp16":
        return torch.float16
    return None
