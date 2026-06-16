"""baselines/runner.py — backward-compat shim.

All implementation lives in ``baselines/_runtime/`` (split per concern):

    precision.py        — model_dtype, autocast_dtype
    data.py             — load_cc_news_v2, load_probes, TokenizedDataset
    eval.py             — eval_cloze_accuracy, pretty_matrix
    logging_setup.py    — setup_logging, TrainerLogCallback
    trainer_factory.py  — standard_trainer, build_training_args, _trainer_kwargs
    runner.py           — Baseline protocol, TrainerRunner

This file simply re-exports the public API so that every existing
``baselines/bN_*.py`` keeps working unchanged (``from runner import …`` resolves
here when the baseline is invoked as ``python baselines/bN_*.py``).
"""
from __future__ import annotations

# Repo root on sys.path so the _runtime package resolves before its first import.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from baselines._runtime import (                             # noqa: E402, F401
    # Public API used by every baseline
    INCAConfig,
    Baseline, TrainerRunner, BaselineRunner,
    TokenizedDataset,
    standard_trainer,
    model_dtype,
    setup_logging,
    LOGGER_NAME,
    # Internal helpers some baselines reach into (B3 uses _trainer_kwargs)
    _trainer_kwargs,
    # Underscore-prefixed back-compat names
    _load_cc_news_v2, _load_probes, _autocast_dtype, _pretty_matrix,
)
