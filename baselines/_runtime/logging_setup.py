"""Logging setup + a TrainerCallback that captures Trainer's per-step logs.

We use Python's standard ``logging`` module with two handlers:

    * stream handler (stdout)             — live progress while you watch.
    * file   handler (out_dir/run.log)    — persistent record per run.

Both receive:
    * messages from ``capsel`` (this runner + every baseline)
    * messages from ``transformers`` (Trainer's per-step loss, deprecation
      warnings, model load info, etc.)

Trainer's per-step output is also forwarded into ``capsel.train`` via
``TrainerLogCallback`` so the loss curve is captured cleanly in our format
(no raw dicts; one well-formatted line per ``logging_steps``).
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

from transformers import TrainerCallback
from transformers.utils import logging as hf_logging


LOGGER_NAME = "capsel"
logger = logging.getLogger(LOGGER_NAME)


def setup_logging(
    out_dir: Path,
    level: int = logging.INFO,
    quiet_hf: bool = True,
) -> logging.Logger:
    """Configure logging for a CL run.

    Parameters
    ----------
    out_dir   : per-run output directory; ``run.log`` is written here.
    level     : verbosity for ``capsel`` (DEBUG | INFO | WARNING).
    quiet_hf  : if True, suppress HuggingFace's noisy deprecation warnings
                and hub download progress bars; transformers logger stays at
                WARNING so real warnings still surface.

    Returns the configured ``capsel`` logger; sub-loggers can be attached via
    ``logging.getLogger("capsel.<name>")``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Our logger ────────────────────────────────────────────────
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False  # don't double-print via root logger

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # ── transformers logger: also write to file ────────────────────
    hf_root = logging.getLogger("transformers")
    for h in list(hf_root.handlers):
        if isinstance(h, logging.FileHandler):
            hf_root.removeHandler(h)
    hf_root.addHandler(file_handler)

    if quiet_hf:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*warmup_ratio.*")
        warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
        hf_logging.set_verbosity_warning()
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except ImportError:
            pass
    else:
        hf_logging.set_verbosity_info()

    hf_logging.enable_explicit_format()
    return logger


class TrainerLogCallback(TrainerCallback):
    """Forward Trainer's per-step log entries into our package logger.

    Trainer normally prints loss/lr/grad_norm as dicts to stdout via its
    ``PrinterCallback``.  We intercept those dicts and re-emit them through
    ``capsel.train`` so they land in both stdout and ``run.log`` with our
    formatting, and so they can be filtered by name.

    Also captures the full step-by-step history so the runner can dump
    ``loss_curve_<pid>.json`` per period without re-parsing log text.
    """

    def __init__(self, period_label: str, period_idx: int):
        self._log = logging.getLogger(f"{LOGGER_NAME}.train")
        self._pid = period_label
        self._idx = period_idx
        self.history: List[Dict[str, Any]] = []   # captured for JSON dump

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        self.history.append({"step": state.global_step, **logs})
        if "loss" in logs:
            grad_norm = logs.get("grad_norm")
            grad_part = f" |g|={grad_norm:.2f}" if grad_norm is not None else ""
            self._log.info(
                "[%s] step=%5d  epoch=%.2f  loss=%.4f  lr=%.2e%s",
                self._pid, state.global_step,
                logs.get("epoch", 0.0), logs["loss"],
                logs.get("learning_rate", 0.0), grad_part,
            )
