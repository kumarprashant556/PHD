"""Unit tests for INCAPlateauDetector  (T1.1 deliverable).

Tests cover:
  - NONE returned before min_epochs guard
  - PERIOD_LEARNED fires on high RIR + loss plateau
  - BLOCK_FULL fires on negligible RIR + plateau + grad-decay
  - BLOCK_FULL fires on negligible RIR + plateau + CKA saturation
  - NONE when only one signal fires (requires consensus)
  - T1.2 timeout: returns PERIOD_LEARNED when RIR ≥ min_rir
  - T1.2 timeout: returns EXHAUSTED when RIR < min_rir
  - GradNormTracker EMA and decay detection
  - LossPlateauTracker plateau detection
  - RIRTracker basic RIR arithmetic
"""

import sys
from pathlib import Path
import pytest

# Make the package importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Phase1.src.inca_plateau import (
    INCAPlateauDetector,
    SaturationEvent,
    GradNormTracker,
    LossPlateauTracker,
    RIRTracker,
)


# ── minimal config stub ────────────────────────────────────────────────────────

class _Cfg:
    rir_threshold       = 0.30
    rir_negligible      = 0.05
    min_rir_for_learned = 0.20
    cka_saturation_threshold = 0.95
    min_epochs_before_grow   = 2
    grad_norm_ema_alpha  = 0.10
    grad_norm_decay_frac = 0.50
    patience             = 3


# ── GradNormTracker ────────────────────────────────────────────────────────────

class TestGradNormTracker:
    def test_initial_state(self):
        t = GradNormTracker()
        assert t.ema == 0.0
        assert not t.is_decayed()

    def test_ema_converges(self):
        t = GradNormTracker(alpha=0.5)
        for _ in range(20):
            t.update(1.0)
        assert abs(t.ema - 1.0) < 1e-3

    def test_decay_detected(self):
        t = GradNormTracker(alpha=0.5, decay_frac=0.50)
        # Drive peak up
        for _ in range(10):
            t.update(10.0)
        peak = t._peak
        # Now feed near-zero
        for _ in range(10):
            t.update(0.0)
        assert t.is_decayed()
        assert t.ema < 0.5 * peak

    def test_no_decay_when_stable(self):
        t = GradNormTracker(alpha=0.5, decay_frac=0.50)
        for _ in range(20):
            t.update(5.0)
        assert not t.is_decayed()

    def test_reset(self):
        t = GradNormTracker()
        t.update(5.0)
        t.reset()
        assert t.ema == 0.0
        assert t._peak == 0.0


# ── LossPlateauTracker ────────────────────────────────────────────────────────

class TestLossPlateauTracker:
    def test_not_plateau_when_window_short(self):
        t = LossPlateauTracker(patience=5, min_delta=1e-3)
        t.update(1.0)
        t.update(0.9)
        assert not t.is_plateau()

    def test_plateau_detected(self):
        t = LossPlateauTracker(patience=3, min_delta=1e-3)
        # All values within 1e-3
        for v in [1.001, 1.000, 1.0005]:
            t.update(v)
        assert t.is_plateau()

    def test_no_plateau_when_improving(self):
        t = LossPlateauTracker(patience=3, min_delta=1e-3)
        for v in [1.0, 0.90, 0.80]:
            t.update(v)
        assert not t.is_plateau()

    def test_reset(self):
        t = LossPlateauTracker(patience=3, min_delta=1e-3)
        for v in [1.0, 1.0, 1.0]:
            t.update(v)
        t.reset()
        assert not t.is_plateau()


# ── RIRTracker ────────────────────────────────────────────────────────────────

class TestRIRTracker:
    def test_zero_before_baseline(self):
        t = RIRTracker(chance=0.25)
        assert t.rir == 0.0

    def test_positive_rir(self):
        t = RIRTracker(chance=0.25)
        t.set_baseline(0.50)
        t.update(0.70)
        # (0.70 - 0.50) / max(0.50, 0.25) = 0.20 / 0.50 = 0.40
        assert abs(t.rir - 0.40) < 1e-6

    def test_chance_floor(self):
        t = RIRTracker(chance=0.25)
        t.set_baseline(0.10)
        t.update(0.30)
        # denom = max(0.10, 0.25) = 0.25;  (0.30 - 0.10) / 0.25 = 0.80
        assert abs(t.rir - 0.80) < 1e-6

    def test_negative_rir(self):
        t = RIRTracker()
        t.set_baseline(0.50)
        t.update(0.40)
        assert t.rir < 0.0

    def test_reset(self):
        t = RIRTracker()
        t.set_baseline(0.50)
        t.update(0.80)
        t.reset()
        assert t.rir == 0.0


# ── INCAPlateauDetector ────────────────────────────────────────────────────────

class TestINCAPlateauDetector:
    def _make_detector(self):
        return INCAPlateauDetector(_Cfg())

    def _fill_plateau(self, det, loss=1.0):
        """Feed enough loss updates to satisfy the plateau window."""
        for _ in range(_Cfg.patience):
            det.loss_tracker.update(loss)

    def _decay_grad(self, det):
        """Drive grad EMA up then down so is_decayed() = True."""
        for _ in range(20):
            det.grad_tracker.update(10.0)
        for _ in range(20):
            det.grad_tracker.update(0.0)

    # ── grokking guard ─────────────────────────────────────────────────
    def test_none_before_min_epochs(self):
        det = self._make_detector()
        det.reset_period(0.5)
        det.update(1.0, 0.8, 1.0, 0.5)
        assert det.check(epoch=1) == SaturationEvent.NONE

    # ── PERIOD_LEARNED ─────────────────────────────────────────────────
    def test_period_learned_fires(self):
        det = self._make_detector()
        det.reset_period(0.50)
        self._fill_plateau(det, loss=1.0)
        # RIR = (0.85 - 0.50) / 0.50 = 0.70 ≥ 0.30
        det.rir_tracker.update(0.85)
        det.grad_tracker.update(1.0)
        det._last_cka = 0.5
        assert det.check(epoch=5) == SaturationEvent.PERIOD_LEARNED

    def test_period_learned_needs_plateau(self):
        """High RIR alone should NOT fire if loss is still dropping."""
        det = self._make_detector()
        det.reset_period(0.50)
        # Only 1 update in window — no plateau yet
        det.loss_tracker.update(1.0)
        det.rir_tracker.update(0.85)
        det._last_cka = 0.5
        assert det.check(epoch=5) == SaturationEvent.NONE

    # ── BLOCK_FULL ─────────────────────────────────────────────────────
    def test_block_full_via_grad_decay(self):
        det = self._make_detector()
        det.reset_period(0.50)
        self._fill_plateau(det)
        det.rir_tracker.update(0.51)   # negligible improvement (≤ 0.05 of baseline)
        det.rir_tracker._baseline = 0.50
        det.rir_tracker._current  = 0.52  # RIR ≈ 0.04 ≤ 0.05
        self._decay_grad(det)
        det._last_cka = 0.5
        assert det.check(epoch=5) == SaturationEvent.BLOCK_FULL

    def test_block_full_via_cka_saturation(self):
        det = self._make_detector()
        det.reset_period(0.50)
        self._fill_plateau(det)
        det.rir_tracker._baseline = 0.50
        det.rir_tracker._current  = 0.52  # RIR ≈ 0.04
        det._last_cka = 0.97   # ≥ 0.95 threshold
        assert det.check(epoch=5) == SaturationEvent.BLOCK_FULL

    def test_block_full_needs_both_rir_and_plateau(self):
        """CKA saturated but RIR is large → should NOT fire BLOCK_FULL."""
        det = self._make_detector()
        det.reset_period(0.50)
        self._fill_plateau(det)
        det.rir_tracker._baseline = 0.50
        det.rir_tracker._current  = 0.85   # RIR = 0.70 > rir_negligible
        det._last_cka = 0.97
        # Should not be BLOCK_FULL (high RIR contradicts block-full condition)
        result = det.check(epoch=5)
        # It may fire PERIOD_LEARNED (high RIR + plateau) but not BLOCK_FULL
        assert result != SaturationEvent.BLOCK_FULL

    # ── T1.2 timeout ───────────────────────────────────────────────────
    def test_timeout_learned(self):
        det = self._make_detector()
        det.reset_period(0.50)
        det.rir_tracker._baseline = 0.50
        det.rir_tracker._current  = 0.75  # RIR = 0.50 ≥ min_rir_for_learned=0.20
        assert det.check_timeout() == SaturationEvent.PERIOD_LEARNED

    def test_timeout_exhausted(self):
        det = self._make_detector()
        det.reset_period(0.50)
        det.rir_tracker._baseline = 0.50
        det.rir_tracker._current  = 0.55  # RIR = 0.10 < 0.20
        assert det.check_timeout() == SaturationEvent.EXHAUSTED

    # ── state_dict ─────────────────────────────────────────────────────
    def test_state_dict_keys(self):
        det = self._make_detector()
        det.reset_period(0.5)
        det.update(1.0, 0.6, 0.5, 0.8)
        sd = det.state_dict()
        for key in ("rir", "grad_ema", "grad_decayed", "loss_plateau", "cka", "eval_steps"):
            assert key in sd, f"Missing key: {key}"
