"""INCA multi-signal consensus saturation detector  (Phase 1, T1.1).

Upgrades the original two-detector design to a four-signal consensus
rule, as specified in Part VII.2 of the CAPSEL memorandum.

Four signals
------------
1. RIR  — Relative Improvement Rate over the period baseline.
     RIR = (score_now - score_0) / max(score_0, chance)
   where score_0 is the score at the start of the current period and
   chance = 0.25 for 4-way MCQ, 0.0 for open-answer.

2. GradNorm (EMA)  — L2 norm of current-block gradients, smoothed by an
   exponential moving average.  Decay is compared to the peak seen since
   the last grow event.

3. CKA drift  — cosine of the representational-similarity between the
   current encoder output and a cached reference set (from CKAMonitor).
   High CKA → stable representation → contributes to block-full.

4. Loss plateau  — training-loss slope over the last ``patience`` eval
   windows, compared to an absolute threshold.

Consensus rules (Part VII.2)
----------------------------
PeriodLearned fires when:
    RIR ≥ rir_threshold  AND  loss plateau holds

BlockFull fires when:
    RIR ≤ rir_negligible  AND  loss plateau holds
    AND  ( grad_norm_decayed  OR  cka_saturated )

Early-stopping relabelling (T1.2)
----------------------------------
When the trainer hits the patience limit without either detector firing:
    if RIR ≥ min_rir_for_learned  →  emit "learned"  (progress is real)
    else                           →  emit "exhausted" → BlockFull path
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional


class SaturationEvent(Enum):
    NONE       = auto()   # no decision yet — continue training
    PERIOD_LEARNED = auto()  # advance to next period (same block)
    BLOCK_FULL     = auto()  # freeze current block, grow new one
    EXHAUSTED      = auto()  # timeout with no RIR — map to block-full path


# ── Grad-norm EMA tracker ─────────────────────────────────────────────────────

class GradNormTracker:
    """Tracks EMA-smoothed gradient L2 norm and detects decay.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (0 < alpha < 1; smaller = slower response).
    decay_frac : float
        The EMA is considered "decayed" when current EMA < decay_frac * peak.
    """

    def __init__(self, alpha: float = 0.10, decay_frac: float = 0.50) -> None:
        self.alpha = alpha
        self.decay_frac = decay_frac
        self._ema: Optional[float] = None
        self._peak: float = 0.0

    def update(self, grad_norm: float) -> float:
        """Update EMA with a new grad-norm observation.  Returns current EMA."""
        if self._ema is None:
            self._ema = grad_norm
        else:
            self._ema = self.alpha * grad_norm + (1 - self.alpha) * self._ema
        if self._ema > self._peak:
            self._peak = self._ema
        return self._ema

    @property
    def ema(self) -> float:
        return self._ema if self._ema is not None else 0.0

    def is_decayed(self) -> bool:
        """True if grad-norm EMA has fallen to < decay_frac × peak."""
        if self._peak < 1e-8:
            return False
        return self.ema < self.decay_frac * self._peak

    def reset(self) -> None:
        self._ema = None
        self._peak = 0.0


# ── Loss-slope tracker ────────────────────────────────────────────────────────

class LossPlateauTracker:
    """Monitors training-loss slope over a sliding window.

    'Plateau' = the absolute difference between the oldest and newest loss
    in the window is less than ``min_delta``.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._window: Deque[float] = deque(maxlen=patience)

    def update(self, loss: float) -> None:
        self._window.append(loss)

    def is_plateau(self) -> bool:
        if len(self._window) < self.patience:
            return False
        return (self._window[0] - self._window[-1]) < self.min_delta

    def reset(self) -> None:
        self._window.clear()


# ── RIR tracker ───────────────────────────────────────────────────────────────

class RIRTracker:
    """Computes RIR = (score_now − score_0) / max(score_0, chance).

    score_0 is set at the start of each period.
    """

    def __init__(self, chance: float = 0.0) -> None:
        self.chance = chance
        self._baseline: Optional[float] = None
        self._current: float = 0.0

    def set_baseline(self, score: float) -> None:
        self._baseline = score
        self._current  = score

    def update(self, score: float) -> float:
        self._current = score
        return self.rir

    @property
    def rir(self) -> float:
        if self._baseline is None:
            return 0.0
        denom = max(self._baseline, self.chance, 1e-8)
        return (self._current - self._baseline) / denom

    def reset(self) -> None:
        self._baseline = None
        self._current  = 0.0


# ── Main consensus detector ───────────────────────────────────────────────────

class INCAPlateauDetector:
    """Multi-signal consensus saturation detector (Phase 1, T1.1).

    Aggregates four signals into two crisp decisions:
      PeriodLearned → advance period
      BlockFull     → freeze-and-grow

    Usage inside train_inca_v3.py::

        detector = INCAPlateauDetector(cfg)

        # at period start:
        detector.reset_period(pre_score)

        # at each k_eval step:
        detector.update(loss, score, grad_norm, cka_monitor, epoch)
        event = detector.check(epoch)
        if event == SaturationEvent.PERIOD_LEARNED:
            ...
    """

    def __init__(self, cfg) -> None:
        self.rir_threshold      = cfg.rir_threshold
        self.rir_negligible     = cfg.rir_negligible
        self.min_rir_for_learned = cfg.min_rir_for_learned
        self.cka_threshold      = cfg.cka_saturation_threshold
        self.min_epochs         = cfg.min_epochs_before_grow

        self.grad_tracker  = GradNormTracker(
            alpha=cfg.grad_norm_ema_alpha,
            decay_frac=cfg.grad_norm_decay_frac,
        )
        self.loss_tracker  = LossPlateauTracker(patience=cfg.patience)
        self.rir_tracker   = RIRTracker(chance=getattr(cfg, "chance", 0.0))

        self._n_eval_steps: int = 0  # eval steps since period start

    # ── period lifecycle ──────────────────────────────────────────────

    def reset_period(self, baseline_score: float) -> None:
        """Call at the beginning of every period (after pre-training eval)."""
        self.rir_tracker.set_baseline(baseline_score)
        self.loss_tracker.reset()
        self.grad_tracker.reset()
        self._n_eval_steps = 0

    def reset_block(self) -> None:
        """Call when a new block is created (after freeze_and_grow)."""
        self.grad_tracker.reset()
        self._n_eval_steps = 0

    # ── signal updates ────────────────────────────────────────────────

    def update(
        self,
        loss: float,
        score: float,
        grad_norm: float,
        cka_value: float,
    ) -> None:
        """Update all four signal trackers with the latest eval-step values."""
        self.loss_tracker.update(loss)
        self.rir_tracker.update(score)
        self.grad_tracker.update(grad_norm)
        self._last_cka = cka_value
        self._n_eval_steps += 1

    # ── decision ──────────────────────────────────────────────────────

    def check(self, epoch: int) -> SaturationEvent:
        """Apply consensus rules and return the appropriate event.

        Called after every ``update``.  Returns NONE until the grokking
        guard (min_epochs_before_grow) has been satisfied.
        """
        if epoch < self.min_epochs:
            return SaturationEvent.NONE

        rir            = self.rir_tracker.rir
        plateau        = self.loss_tracker.is_plateau()
        grad_decayed   = self.grad_tracker.is_decayed()
        cka_stable     = getattr(self, "_last_cka", 0.0) >= self.cka_threshold

        # PeriodLearned: strong improvement + loss converged
        if rir >= self.rir_threshold and plateau:
            return SaturationEvent.PERIOD_LEARNED

        # BlockFull: no improvement + loss converged + capacity exhausted
        if rir <= self.rir_negligible and plateau and (grad_decayed or cka_stable):
            return SaturationEvent.BLOCK_FULL

        return SaturationEvent.NONE

    def check_timeout(self) -> SaturationEvent:
        """[T1.2] Called when patience limit is hit without a decision.

        If RIR ≥ min_rir_for_learned the model made real progress → treat as
        PERIOD_LEARNED.  Otherwise the block is exhausted → BLOCK_FULL path.
        """
        if self.rir_tracker.rir >= self.min_rir_for_learned:
            return SaturationEvent.PERIOD_LEARNED
        return SaturationEvent.EXHAUSTED

    # ── diagnostics ───────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "rir":         self.rir_tracker.rir,
            "grad_ema":    self.grad_tracker.ema,
            "grad_decayed": self.grad_tracker.is_decayed(),
            "loss_plateau": self.loss_tracker.is_plateau(),
            "cka":         getattr(self, "_last_cka", None),
            "eval_steps":  self._n_eval_steps,
        }
