"""INCA multi-signal consensus saturation detector  (Phase 1, T1.1).

Five signals
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

4. Loss plateau  — training-loss range over the last ``patience`` eval
   windows, compared to an absolute threshold.  Used only to gate the
   timeout counter in the trainer; NOT used in the primary consensus rules.

5. Metric early stop  — mirrors the Transformers EarlyStoppingCallback:
   tracks the best eval F1 seen so far and counts consecutive checks with
   no improvement ≥ ``early_stop_min_delta``.  This is the primary
   convergence gate replacing loss plateau in check().

Consensus rules
---------------
PeriodLearned fires when:
    metric_stalled  AND  RIR ≥ rir_threshold

BlockFull fires when:
    metric_stalled  AND  RIR ≤ rir_negligible
    AND  ( grad_norm_decayed  OR  cka_saturated )

Early-stopping relabelling (T1.2)
----------------------------------
When the trainer hits the patience_timeout limit without either detector
firing (checked only when loss is flat):
    if RIR ≥ min_rir_for_learned  →  emit "learned"
    else                           →  emit "exhausted" → BlockFull path
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Deque, Optional


class SaturationEvent(Enum):
    NONE         = auto()   # no decision yet — continue training
    PERIOD_LEARNED = auto()  # advance to next period (same block)
    BLOCK_FULL     = auto()  # freeze current block, grow new one
    EXHAUSTED      = auto()  # timeout with no RIR — map to block-full path


# ── Grad-norm EMA tracker ─────────────────────────────────────────────────────

class GradNormTracker:
    """EMA-smoothed gradient L2 norm; detects when norm decays below a fraction of its peak."""

    def __init__(self, alpha: float = 0.10, decay_frac: float = 0.50) -> None:
        self.alpha = alpha
        self.decay_frac = decay_frac
        self._ema: Optional[float] = None
        self._peak: float = 0.0

    def update(self, grad_norm: float) -> float:
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
        if self._peak < 1e-8:
            return False
        return self.ema < self.decay_frac * self._peak

    def reset(self) -> None:
        self._ema = None
        self._peak = 0.0


# ── Loss-plateau tracker ──────────────────────────────────────────────────────

class LossPlateauTracker:
    """Monitors training-loss stability over a sliding window.

    Uses max-min range (not endpoint diff) to avoid being fooled by
    epoch-boundary spikes sitting in the middle of the window.
    Used only to gate the timeout counter; not part of the primary consensus.
    """

    def __init__(self, patience: int = 5, min_delta: float = 5e-3) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._window: Deque[float] = deque(maxlen=patience)

    def update(self, loss: float) -> None:
        self._window.append(loss)

    def is_plateau(self) -> bool:
        if len(self._window) < self.patience:
            return False
        return (max(self._window) - min(self._window)) < self.min_delta

    def reset(self) -> None:
        self._window.clear()


# ── Metric early-stop tracker ─────────────────────────────────────────────────

class MetricEarlyStop:
    """Fires when the eval metric (F1) stops improving — mirrors EarlyStoppingCallback.

    Tracks the best score seen so far.  Each eval that fails to improve by at
    least ``min_delta`` increments a counter; the counter resets on any genuine
    improvement.  ``should_stop()`` returns True once the counter reaches
    ``patience``.

    Advantages over loss-plateau for the primary convergence signal:
    - Tracks task performance (F1) directly, not a proxy (loss)
    - Immune to epoch-boundary loss spikes
    - Handles cases where loss ticks down but F1 has converged
    """

    def __init__(self, patience: int = 5, min_delta: float = 5e-3) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best:       float = -float("inf")
        self._no_improve: int   = 0

    def update(self, score: float) -> None:
        if score > self._best + self.min_delta:
            self._best       = score
            self._no_improve = 0
        else:
            self._no_improve += 1

    def should_stop(self) -> bool:
        return self._no_improve >= self.patience

    @property
    def best(self) -> float:
        return self._best

    @property
    def no_improve_count(self) -> int:
        return self._no_improve

    def reset(self) -> None:
        self._best       = -float("inf")
        self._no_improve = 0


# ── RIR tracker ───────────────────────────────────────────────────────────────

class RIRTracker:
    """Computes RIR = (score_now − score_0) / max(score_0, chance)."""

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

    Primary convergence gate: MetricEarlyStop (F1-based, mirrors EarlyStoppingCallback).
    Secondary (timeout gating only): LossPlateauTracker.

    Usage::

        detector = INCAPlateauDetector(cfg)

        # at period start:
        detector.reset_period(pre_score)

        # at each k_eval step:
        detector.update(loss, score, grad_norm, cka_value)
        event = detector.check(epoch)
    """

    def __init__(self, cfg) -> None:
        self.rir_threshold       = cfg.rir_threshold
        self.rir_negligible      = cfg.rir_negligible
        self.min_rir_for_learned = cfg.min_rir_for_learned
        self.cka_threshold       = cfg.cka_saturation_threshold
        self.min_epochs          = cfg.min_epochs_before_grow

        self.grad_tracker  = GradNormTracker(
            alpha=cfg.grad_norm_ema_alpha,
            decay_frac=cfg.grad_norm_decay_frac,
        )
        self.loss_tracker  = LossPlateauTracker(
            patience=cfg.patience,
            min_delta=getattr(cfg, "loss_plateau_min_delta", 5e-3),
        )
        self.rir_tracker   = RIRTracker(chance=getattr(cfg, "chance", 0.0))
        self.early_stop    = MetricEarlyStop(
            patience=getattr(cfg, "early_stop_patience", 5),
            min_delta=getattr(cfg, "early_stop_min_delta", 5e-3),
        )

        self._n_eval_steps: int = 0

    # ── period / block lifecycle ──────────────────────────────────────

    def reset_period(self, baseline_score: float) -> None:
        """Call at the beginning of every period (after pre-period eval)."""
        self.rir_tracker.set_baseline(baseline_score)
        self.loss_tracker.reset()
        self.grad_tracker.reset()
        self.early_stop.reset()
        self._n_eval_steps = 0

    def reset_block(self) -> None:
        """Call when a new block is created (after freeze_and_grow)."""
        self.grad_tracker.reset()
        self.early_stop.reset()
        self._n_eval_steps = 0

    # ── signal updates ────────────────────────────────────────────────

    def update(
        self,
        loss: float,
        score: float,
        grad_norm: float,
        cka_value: float,
    ) -> None:
        """Update all trackers with the latest eval-step values."""
        self.loss_tracker.update(loss)
        self.rir_tracker.update(score)
        self.grad_tracker.update(grad_norm)
        self.early_stop.update(score)
        self._last_cka = cka_value
        self._n_eval_steps += 1

    # ── decision ──────────────────────────────────────────────────────

    def check(self, epoch: int) -> SaturationEvent:
        """Apply consensus rules; returns NONE until grokking guard is satisfied."""
        if epoch < self.min_epochs:
            return SaturationEvent.NONE

        rir          = self.rir_tracker.rir
        stalled      = self.early_stop.should_stop()   # primary convergence gate
        grad_decayed = self.grad_tracker.is_decayed()
        cka_stable   = getattr(self, "_last_cka", 0.0) >= self.cka_threshold

        # PeriodLearned: F1 stalled + block made meaningful progress
        if stalled and rir >= self.rir_threshold:
            return SaturationEvent.PERIOD_LEARNED

        # BlockFull: F1 stalled + block made no progress + capacity signals confirm
        if stalled and rir <= self.rir_negligible and (grad_decayed or cka_stable):
            return SaturationEvent.BLOCK_FULL

        return SaturationEvent.NONE

    @property
    def loss_plateau(self) -> bool:
        """True when the loss plateau condition is met. Used to gate timeout counter."""
        return self.loss_tracker.is_plateau()

    def check_timeout(self) -> SaturationEvent:
        """[T1.2] Called when patience_timeout is hit (loss flat but no signal fired).

        High RIR → PERIOD_LEARNED (progress was real, block not exhausted).
        Low RIR  → EXHAUSTED → freeze-and-grow.
        """
        if self.rir_tracker.rir >= self.min_rir_for_learned:
            return SaturationEvent.PERIOD_LEARNED
        return SaturationEvent.EXHAUSTED

    # ── diagnostics ───────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "rir":            self.rir_tracker.rir,
            "grad_ema":       self.grad_tracker.ema,
            "grad_decayed":   self.grad_tracker.is_decayed(),
            "loss_plateau":   self.loss_tracker.is_plateau(),
            "early_stop_best":       self.early_stop.best,
            "early_stop_no_improve": self.early_stop.no_improve_count,
            "metric_stalled": self.early_stop.should_stop(),
            "cka":            getattr(self, "_last_cka", None),
            "eval_steps":     self._n_eval_steps,
        }
