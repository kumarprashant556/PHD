"""Phase 0 shared configuration.

Baselines load this via :func:`load_config` which reads a YAML/JSON file from
``Phase0/configs/`` and overlays CLI flags. Every baseline uses the same
:class:`Phase0Config` so that runs are reproducible and comparable.

Field-by-field semantics:

* ``dataset``             – folder name under ``Phase0/data/processed/``
                            (``temporalwiki``, ``cc_news`` etc).
* ``model_name``          – HuggingFace model id; defaults to ``distilgpt2``.
* ``max_periods``         – number of temporal periods consumed.
* ``max_docs_per_period`` – soft cap per period; 0 means "no cap".
* ``epochs_per_period``   – fine-tune epochs (or equivalent) per period.
* ``batch_size``          – train and eval batch size.
* ``lr``                  – AdamW learning rate.
* ``weight_decay``        – AdamW weight decay.
* ``max_seq_len``         – truncation length for causal LM input.
* ``ppl_eval_samples``    – cap on eval set for PPL computation.
* ``ppl_eval_frac``       – train / eval split fraction (eval side).
* ``probe_max``           – cap on probes used for probe accuracy.
* ``seed``                – reproducibility seed.
* ``results_root``        – where each run stores its artefacts.

Baseline-specific hyperparameters (lambda for EWC, LoRA rank for B5, prompt
length for L2P, etc.) live on dataclasses in the individual baseline modules,
keeping this shared config free of method-specific clutter.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Phase0Config:
    # dataset + model
    dataset: str = "realtime_qa"
    model_name: str = "google/flan-t5-base"
    model_type: str = "seq2seq"       # "seq2seq" (T5-style) | "causal" (GPT-style)
    max_periods: int = 4
    max_docs_per_period: int = 0      # 0 / None → no cap; applied by loader
    max_train_probes: int = 0         # 0 → no cap; caps seq2seq training probes per period

    # training
    epochs_per_period: int = 3
    batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_seq_len: int = 256            # encoder input length (question + context)
    max_answer_len: int = 32          # decoder target length (answer tokens)
    max_new_tokens: int = 32          # max tokens to generate during eval
    grad_accum_steps: int = 1         # gradient accumulation (effective batch = batch_size × steps)
    warmup_ratio: float = 0.06        # fraction of total steps used for LR linear warmup
    log_every_n_steps: int = 50       # write loss to loss_curve.csv every N optimiser steps
    gradient_checkpointing: bool = False  # recompute activations to save memory (slower)
    use_adafactor: bool = False           # Adafactor optimizer (lower mem than AdamW; good for xl)

    # eval
    ppl_eval_samples: int = 300
    ppl_eval_frac: float = 0.20
    probe_max: int = 200
    eval_mode: str = "combined"       # "em" | "f1" | "combined"
    em_weight: float = 0.4            # weight for EM in combined score (seq2seq)
    f1_weight: float = 0.6            # weight for F1 in combined score (seq2seq)
    ppl_weight: float = 0.4           # weight for PPL-score in combined (causal)
    probe_weight: float = 0.6         # weight for probe_acc in combined (causal)
    ppl_decay: float = 0.3            # shaping constant for PPL→score (causal)

    # housekeeping
    seed: int = 42
    results_root: str = ""            # filled in by harness if empty
    device: str = ""                  # auto-select if empty
    notes: str = ""                   # free-form run label


# ── YAML/JSON loader ─────────────────────────────────────────────────────────

def _read_cfg_file(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "PyYAML is required to load YAML configs. "
                "Use a .json config, or pip install pyyaml --break-system-packages."
            ) from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Phase0Config:
    """Load a Phase0Config from file, then apply CLI overrides.

    Unknown keys in the file or the overrides are silently ignored so the
    common config stays stable even when a baseline extends it.
    """
    data: Dict[str, Any] = {}
    if config_path:
        data.update(_read_cfg_file(Path(config_path)))

    # shallow flatten: we allow nested sections in the YAML (training: {...},
    # perplexity: {...}) so the file is readable, but the dataclass is flat.
    flat: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v

    if overrides:
        flat.update({k: v for k, v in overrides.items() if v is not None})

    allowed = {f.name for f in Phase0Config.__dataclass_fields__.values()}
    filtered = {k: v for k, v in flat.items() if k in allowed}
    return Phase0Config(**filtered)


def snapshot_config(cfg: Phase0Config, extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a dict snapshot of the config for ``config.snapshot.json``."""
    out = asdict(cfg)
    if extras:
        out["baseline_extras"] = extras
    return out


# ── Device selection ────────────────────────────────────────────────────────

def auto_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str):
    """Return the appropriate torch dtype for a given device.

    CUDA  → float16   (fast, numerically stable on CUDA with automatic mixed precision)
    MPS   → float32   (float16 overflows to NaN on Metal; float32 is the safe choice)
    CPU   → float32
    """
    import torch
    if "cuda" in device:
        return torch.float16
    return torch.float32


def ensure_results_dir(cfg: Phase0Config, baseline_id: str) -> Path:
    """Ensure ``results/<baseline_id>/`` exists and return it.

    ``results_root`` defaults to ``Phase0/results`` relative to this file.
    """
    if cfg.results_root:
        root = Path(cfg.results_root)
    else:
        root = Path(__file__).resolve().parent.parent / "results"
    out = root / baseline_id
    out.mkdir(parents=True, exist_ok=True)
    return out
