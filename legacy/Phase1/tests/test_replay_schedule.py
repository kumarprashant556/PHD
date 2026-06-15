"""Unit tests for INCAReplayBuffer  (T1.4 deliverable).

Tests cover:
  - add_period stores items, respecting max_size_per_period cap
  - sample returns exactly n items (or fewer if buffer is small)
  - Phase A (epoch < n_revise): uniform sampling (all items reachable)
  - Phase B (epoch ≥ n_revise): p_hard / p_easy / p_mid proportions
  - update_losses refreshes stored loss values
  - clear_period removes a single period
  - clear_all empties the buffer
  - stats() returns per-period counts
  - all_items() returns every stored item
  - __len__ counts total entries
  - Multiple periods are sampled from jointly
"""

import sys
from pathlib import Path
import random

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Phase1.src.inca_replay import INCAReplayBuffer


# ── fixtures ──────────────────────────────────────────────────────────────────

def _items(n: int, prefix: str = "item") -> list:
    return [{"id": f"{prefix}_{i}", "question": f"q{i}", "answer": str(i)} for i in range(n)]


# ── basic CRUD ────────────────────────────────────────────────────────────────

class TestBasicCRUD:
    def test_add_and_len(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        buf.add_period("p1", _items(50))
        assert len(buf) == 50

    def test_max_size_cap(self):
        buf = INCAReplayBuffer(max_size_per_period=30)
        buf.add_period("p1", _items(100))
        assert len(buf) == 30

    def test_multiple_periods(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        buf.add_period("p1", _items(40))
        buf.add_period("p2", _items(60))
        assert len(buf) == 100

    def test_all_items(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        buf.add_period("p1", _items(10))
        assert len(buf.all_items()) == 10

    def test_clear_period(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        buf.add_period("p1", _items(20))
        buf.add_period("p2", _items(20))
        buf.clear_period("p1")
        assert len(buf) == 20
        assert "p1" not in buf.periods

    def test_clear_all(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        buf.add_period("p1", _items(20))
        buf.clear_all()
        assert len(buf) == 0

    def test_stats(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        buf.add_period("p1", _items(15))
        buf.add_period("p2", _items(25))
        stats = buf.stats()
        assert stats["p1"] == 15
        assert stats["p2"] == 25

    def test_periods_property(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        buf.add_period("p1", _items(5))
        buf.add_period("p2", _items(5))
        assert set(buf.periods) == {"p1", "p2"}


# ── sampling ──────────────────────────────────────────────────────────────────

class TestSampling:
    def test_sample_returns_correct_count(self):
        buf = INCAReplayBuffer(max_size_per_period=100, n_revise=3)
        buf.add_period("p1", _items(50))
        result = buf.sample(10, epoch=1)
        assert len(result) == 10

    def test_sample_empty_buffer(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        assert buf.sample(10, epoch=1) == []

    def test_sample_more_than_buffer(self):
        buf = INCAReplayBuffer(max_size_per_period=100, n_revise=3)
        buf.add_period("p1", _items(5))
        result = buf.sample(20, epoch=1)
        assert len(result) == 5   # capped at buffer size

    def test_phase_a_uniform_reachability(self):
        """In Phase A every item should appear with non-negligible probability."""
        random.seed(0)
        buf = INCAReplayBuffer(max_size_per_period=100, n_revise=5)
        items = _items(20)
        buf.add_period("p1", items)

        seen_ids = set()
        for _ in range(200):
            batch = buf.sample(5, epoch=1)   # epoch < n_revise → Phase A
            seen_ids.update(it["id"] for it in batch)

        # After 200 draws of 5 from 20, expect to see all 20
        assert len(seen_ids) == 20

    def test_phase_b_hard_items_dominate(self):
        """In Phase B, items with highest loss should appear most often."""
        random.seed(42)
        n = 30
        buf = INCAReplayBuffer(
            max_size_per_period=100, n_revise=1,
            p_hard=0.70, p_easy=0.20, p_mid=0.10,
        )
        items = _items(n)
        buf.add_period("p1", items)

        # Assign losses: items 0..9 get loss=10.0 (hard), rest get 0.1
        losses = [10.0 if i < 10 else 0.1 for i in range(n)]
        buf.update_losses("p1", items, losses)

        hard_ids = {f"item_{i}" for i in range(10)}
        counts = {iid: 0 for iid in hard_ids}

        n_draws = 500
        for _ in range(n_draws):
            batch = buf.sample(10, epoch=5)   # epoch ≥ n_revise → Phase B
            for it in batch:
                if it["id"] in hard_ids:
                    counts[it["id"]] += 1

        total_hard = sum(counts.values())
        total_samples = n_draws * 10
        hard_frac = total_hard / total_samples
        # With p_hard=0.70, expect ~70% of draws to come from hard pool (10/30 items)
        # Hard pool = top third = items 0..9; so ~70% of draws should be from them
        assert hard_frac > 0.40, f"Hard fraction too low: {hard_frac:.3f}"


# ── update_losses ─────────────────────────────────────────────────────────────

class TestUpdateLosses:
    def test_update_losses_applied(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        items = _items(10)
        buf.add_period("p1", items)
        losses = [float(i) for i in range(10)]
        buf.update_losses("p1", items, losses)

        # Inspect internal store: entries should have updated losses
        store = buf._store["p1"]
        actual_losses = {e.item["id"]: e.loss for e in store}
        for i, it in enumerate(items):
            if it["id"] in actual_losses:
                assert actual_losses[it["id"]] == pytest.approx(float(i))

    def test_update_losses_missing_period_silent(self):
        buf = INCAReplayBuffer(max_size_per_period=100)
        # Should not raise even if period not present
        buf.update_losses("nonexistent", _items(5), [1.0] * 5)


# ── proportion invariants ─────────────────────────────────────────────────────

class TestProportionInvariants:
    def test_proportions_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            INCAReplayBuffer(p_hard=0.5, p_easy=0.3, p_mid=0.3)  # sums to 1.1

    def test_valid_proportions_accepted(self):
        buf = INCAReplayBuffer(p_hard=0.60, p_easy=0.30, p_mid=0.10)
        assert buf.p_hard == 0.60
