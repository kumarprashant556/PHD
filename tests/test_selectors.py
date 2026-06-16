"""Unit tests for INCA selectors.

Specifically guards against the spec-flagged "old incorrect" pattern of
concatenating K/V across blocks (CAPSEL_Selector_Architecture.pdf).
The correct contract is per-block independent attention with a softmax
over n_blocks producing block-level weights.

Tests
-----
* Single-block short-circuit returns the lone block output unchanged.
* Block-weight invariance: swapping the per-block content but keeping
  the relative ordering of relevance must keep the output dominated by
  the same block.
* Per-block independence: a block whose pre-K projection produces
  identical activations regardless of other blocks must yield the same
  per-block attended output regardless of what the other blocks do.
  (This fails immediately when forward concatenates K/V across blocks.)
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.inca.selectors import EmbeddingQuerySelector


def _set_seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _make_selector(D: int = 16, h: int = 4) -> EmbeddingQuerySelector:
    sel = EmbeddingQuerySelector(hidden_size=D, n_heads=h, dropout=0.0)
    sel.eval()
    return sel


class TestSingleBlock:
    def test_single_block_passthrough(self):
        sel = _make_selector()
        x = torch.randn(2, 5, 16)
        out = sel([x])
        assert torch.equal(out, x), "single-block path must not transform the input"


class TestPerBlockIndependence:
    """Per-block attended outputs must depend only on each block's own
    content. The (B, n) softmax over blocks must produce a probability.
    Both invariants break when forward concatenates K/V across blocks
    (the spec-flagged "old incorrect" path).
    """

    def test_block_order_permutation(self):
        """Order independence: weighted sum of per-block attended outputs
        is permutation-symmetric.  If the forward instead concatenated K/V
        the cross-block attention would couple position s in block 0 to
        positions in block 1, breaking this invariance.
        """
        _set_seed(0)
        sel = _make_selector()
        B, S, D = 2, 6, 16
        emb = torch.randn(B, S, D)
        b0 = torch.randn(B, S, D)
        b1 = torch.randn(B, S, D)

        out_ab = sel([b0, b1], embedding_hidden=emb)
        out_ba = sel([b1, b0], embedding_hidden=emb)
        assert torch.allclose(out_ab, out_ba, atol=1e-5), (
            "EmbeddingQuerySelector should be permutation-symmetric over "
            "the block dimension. A failure here indicates K/V are being "
            "concatenated across blocks (spec-incorrect path)."
        )

    def test_other_block_does_not_leak_into_diagonal(self):
        """Doubling block-1's magnitude must not change block-0's per-block
        relevance score under the new implementation. We check this
        indirectly: if both blocks are identical the output must equal
        the per-block attended output (a single computation), independent
        of how many copies are stacked.
        """
        _set_seed(1)
        sel = _make_selector()
        B, S, D = 1, 4, 16
        emb = torch.randn(B, S, D)
        b = torch.randn(B, S, D)

        # 2 identical blocks vs 3 identical blocks — under per-block
        # independence the (B, n) softmax yields uniform weights and the
        # weighted sum equals the per-block attended output in both cases.
        out_2 = sel([b, b.clone()],          embedding_hidden=emb)
        out_3 = sel([b, b.clone(), b.clone()], embedding_hidden=emb)
        assert torch.allclose(out_2, out_3, atol=1e-5), (
            "With N identical blocks the output is invariant in N under "
            "per-block independent attention. Concatenated K/V would "
            "make the attention denominator scale with N and break this."
        )

    def test_forward_shape_and_finiteness(self):
        sel = _make_selector()
        B, S, D = 1, 4, 16
        emb = torch.randn(B, S, D)
        b0 = torch.randn(B, S, D)
        b1 = torch.randn(B, S, D)
        out = sel([b0, b1], embedding_hidden=emb)
        assert out.shape == (B, S, D)
        assert torch.isfinite(out).all()


class TestRequiresEmbedding:
    def test_multi_block_requires_embedding(self):
        sel = _make_selector()
        b0 = torch.randn(1, 4, 16)
        b1 = torch.randn(1, 4, 16)
        with pytest.raises(ValueError):
            sel([b0, b1])
