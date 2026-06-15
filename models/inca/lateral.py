"""Lateral adapter stub  (capsel/models/inca/lateral.py)

Phase 2 implementation target.  Rank-r adapters from a frozen block's
output to the current trainable block's input.

Design (from CAPSEL memorandum Part IV):
  output = hidden + tanh(alpha) * cross_attention(hidden, frozen_block_out)
  alpha initialised to 0.0 (function-preserving at attach time).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class LateralAdapter(nn.Module):
    """Controlled blend from a frozen block into the current trainable block.

    Parameters
    ----------
    hidden_size : model d_model
    rank        : adapter rank r (4, 8, or 16 — ablated in E-SCOPE)
    """

    def __init__(self, hidden_size: int, rank: int = 8) -> None:
        super().__init__()
        self.rank = rank
        # Low-rank projection: frozen_out → rank → hidden_size
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up   = nn.Linear(rank, hidden_size, bias=False)
        # Gating scalar: tanh(alpha) starts at 0 (identity init)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Zero-init up projection → function-preserving at attach time
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        hidden: torch.Tensor,          # (B, S, D) — current block input
        frozen_out: torch.Tensor,      # (B, S, D) — frozen block output
    ) -> torch.Tensor:
        """Return hidden + tanh(alpha) * adapter(frozen_out)."""
        adapter_signal = self.up(torch.tanh(self.down(frozen_out)))
        return hidden + torch.tanh(self.alpha) * adapter_signal

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha.item():.4f}"
