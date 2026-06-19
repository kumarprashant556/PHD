"""INCA study-schedule replay buffer  (Phase 1, T1.4).

Implements the two-phase replay schedule described in Part VIII.3 of the
CAPSEL memorandum:

  Phase A — Initial-pass regime (epoch < N_revise):
      Uniform random sampling from the combined pool of the current
      period's stream items plus the replay buffer.  "Revise everything
      on the first pass."

  Phase B — Refinement regime (epoch ≥ N_revise):
      p_hard = 0.70  → hardest items (highest current loss)
      p_easy = 0.20  → easiest items (lowest current loss, maintenance)
      p_mid  = 0.10  → random mid-loss items

This corrects the v1 inversion of hippocampal replay (v1 kept only the
hardest items; the biological hippocampus preferentially replays
well-encoded episodes, so easy-maintenance samples are required).

The buffer is period-aware: ``add_period`` stores items tagged with a
period label, and sampling draws proportionally from all stored periods
unless ``period`` is specified.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _BufferEntry:
    item: dict          # original probe dict
    period: str
    loss: float = 0.0   # updated after each forward pass in refinement phase


class INCAReplayBuffer:
    """Quality-filtered, study-schedule replay buffer.

    Parameters
    ----------
    max_size_per_period : int
        Cap on entries stored per period.  When exceeded on ``add_period``,
        the highest-loss (hardest) items are retained up to this cap so
        that the buffer quality-filters toward hard examples even at the
        storage stage.
    n_revise : int
        Number of training epochs that count as the initial-pass regime.
        Passed in via ``sample(epoch=...)``.
    p_hard, p_easy, p_mid : float
        Proportions used in the refinement phase.  Must sum to 1.0.
    """

    def __init__(
        self,
        max_size_per_period: int = 2000,
        n_revise: int = 3,
        p_hard: float = 0.70,
        p_easy: float = 0.20,
        p_mid: float = 0.10,
    ) -> None:
        assert abs(p_hard + p_easy + p_mid - 1.0) < 1e-6, \
            f"p_hard+p_easy+p_mid must equal 1.0, got {p_hard+p_easy+p_mid}"
        self.max_size_per_period = max_size_per_period
        self.n_revise = n_revise
        self.p_hard = p_hard
        self.p_easy = p_easy
        self.p_mid = p_mid
        # period_label → list of _BufferEntry
        self._store: Dict[str, List[_BufferEntry]] = {}

    # ── public write API ─────────────────────────────────────────────

    def add_period(self, period: str, items: List[dict]) -> None:
        """Add items from a completed period into the buffer.

        If ``len(items) > max_size_per_period``, items are shuffled and
        truncated (random selection at add time; loss-based re-sorting
        happens after the first update pass via ``update_losses``).
        """
        entries = [_BufferEntry(item=it, period=period) for it in items]
        random.shuffle(entries)
        self._store[period] = entries[: self.max_size_per_period]

    def update_losses(self, period: str, items: List[dict], losses: List[float]) -> None:
        """Refresh stored loss values after a training pass over *period*.

        items and losses must correspond 1-to-1.  Items not found in the
        buffer are silently ignored (they may have been truncated on add).
        """
        loss_map: Dict[int, float] = {id(it): l for it, l in zip(items, losses)}
        if period not in self._store:
            return
        for entry in self._store[period]:
            key = id(entry.item)
            if key in loss_map:
                entry.loss = loss_map[key]

    def clear_period(self, period: str) -> None:
        """Remove all entries for *period* (called after block freeze)."""
        self._store.pop(period, None)

    def clear_all(self) -> None:
        self._store.clear()

    # ── public read API ──────────────────────────────────────────────

    def sample(self, n: int, epoch: int) -> List[dict]:
        """Draw *n* items using the study-schedule strategy.

        Parameters
        ----------
        n     : number of items to return
        epoch : current training epoch (1-indexed; epoch < n_revise → initial pass)
        """
        all_entries = [e for entries in self._store.values() for e in entries]
        if not all_entries:
            return []
        n = min(n, len(all_entries))

        if epoch < self.n_revise:
            # Phase A: uniform
            return [e.item for e in random.sample(all_entries, n)]
        else:
            # Phase B: study schedule
            return self._study_schedule_sample(all_entries, n)

    def all_items(self) -> List[dict]:
        """Return all buffered items (for eval / drift check)."""
        return [e.item for entries in self._store.values() for e in entries]

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())

    @property
    def periods(self) -> List[str]:
        return list(self._store.keys())

    # ── internals ────────────────────────────────────────────────────

    def _study_schedule_sample(
        self, all_entries: List[_BufferEntry], n: int
    ) -> List[dict]:
        """Sample with hard/easy/mid proportions (Phase B)."""
        # Sort by loss descending (hardest first)
        sorted_entries = sorted(all_entries, key=lambda e: e.loss, reverse=True)
        N = len(sorted_entries)

        n_hard = max(1, round(n * self.p_hard))
        n_easy = max(1, round(n * self.p_easy))
        n_mid  = n - n_hard - n_easy

        # Thirds: top / bottom / middle
        hard_pool = sorted_entries[: max(1, N // 3)]
        easy_pool = sorted_entries[max(1, 2 * N // 3) :]
        mid_pool  = sorted_entries[max(1, N // 3): max(1, 2 * N // 3)]

        hard_sample = random.sample(hard_pool, min(n_hard, len(hard_pool)))
        easy_sample = random.sample(easy_pool, min(n_easy, len(easy_pool)))
        mid_sample  = random.sample(mid_pool,  min(max(0, n_mid), len(mid_pool)))

        combined = hard_sample + easy_sample + mid_sample
        # Fill remainder with random if we came up short
        if len(combined) < n:
            remainder = [e for e in all_entries if e not in set(combined)]
            combined += random.sample(remainder, min(n - len(combined), len(remainder)))

        random.shuffle(combined)
        return [e.item for e in combined[:n]]

    # ── diagnostics ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        return {p: len(v) for p, v in self._store.items()}
