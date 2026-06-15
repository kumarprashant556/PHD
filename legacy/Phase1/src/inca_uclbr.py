"""UCLBR — Uncertainty-Calibrated Load-Balanced Router.

Three components stacked into a single selector module:

  1. Read-ME pre-gate
     ─────────────────
     A lightweight two-layer MLP scores each block's mean-pooled output
     for relevance.  Blocks below a soft relevance threshold are zeroed
     out before full routing, preventing irrelevant frozen blocks from
     polluting the aggregation.  The gate is sigmoid-activated so it
     produces a continuous mask (not hard top-k), keeping gradients alive.

  2. DeepSeek auxiliary-loss-free load balancing
     ──────────────────────────────────────────────
     A per-block bias term b_i is stored as a non-gradient buffer and
     updated online after every forward pass:

         b_i ← b_i + η_b · (1/n − f_i)

     where f_i is the fraction of the current batch's routing weight that
     went to block i and η_b is the bias learning rate (default 1e-3).
     No auxiliary loss is added to the training objective (pure DeepSeek
     style).  The bias nudges under-used blocks up and over-used blocks
     down until load is roughly balanced.

  3. Uncertainty-calibrated confidence
     ────────────────────────────────────
     The entropy of the routing distribution H = −Σ w_i log w_i is
     normalised to [0, 1] by dividing by log(n).  Confidence c = 1 − H̃.

     Final weights interpolate between the learned routing and uniform:

         w_final = c · w_router + (1 − c) · (1/n)

     When the router is uncertain (high entropy → low c) the output falls
     back toward a uniform mixture of all blocks — safer than committing
     to a bad routing decision.

Usage in E-ROUTE ablation
─────────────────────────
  selector: "uclbr"  in YAML config.

  Config keys:
    uclbr_pre_gate_hidden  (int, default 64)   — pre-gate MLP hidden dim
    uclbr_top_k            (int, default 0)    — 0 = soft gate, >0 = keep top-k blocks
    uclbr_bias_lr          (float, default 1e-3) — load-balance bias update rate
    uclbr_heads            (int, default 4)    — routing attention heads

References
──────────
  DeepSeek-MoE (Dai et al., 2024) — auxiliary-loss-free load balancing.
  Read-ME (Zhao et al., 2024)     — pre-gating for efficient MoE routing.
  Uncertainty routing inspiration — Guo et al. calibration (ICML 2017).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UCLBRSelector(nn.Module):
    """Uncertainty-Calibrated Load-Balanced Router.

    Parameters
    ----------
    hidden_size : int
        Encoder hidden dimension D.
    pre_gate_hidden : int
        Hidden units in the Read-ME pre-gate MLP.
    n_heads : int
        Attention heads for the main routing projection.
    bias_lr : float
        Online update rate for the load-balance bias terms.
    top_k : int
        If > 0, only the top-k blocks by pre-gate score are kept (hard
        sparsity).  If 0 (default), soft sigmoid gating is used instead.
    """

    def __init__(
        self,
        hidden_size: int,
        pre_gate_hidden: int = 64,
        n_heads: int = 4,
        bias_lr: float = 1e-3,
        top_k: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size    = hidden_size
        self.n_heads        = n_heads
        self.head_dim       = hidden_size // n_heads
        self.scale          = self.head_dim ** -0.5
        self.bias_lr        = bias_lr
        self.top_k          = top_k

        # ── 1. Read-ME pre-gate ────────────────────────────────────────
        # Input: mean-pooled block output (B, D)
        # Output: scalar relevance score in (0, 1)
        self.pre_gate = nn.Sequential(
            nn.Linear(hidden_size, pre_gate_hidden),
            nn.ReLU(),
            nn.Linear(pre_gate_hidden, 1),
            nn.Sigmoid(),
        )

        # ── 2. Routing projections (multi-head) ───────────────────────
        # Q from original embeddings (no Q projection — same principle as S-QKV)
        # K, V from block outputs
        self.k_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Identity init — function-preserving at grow time
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

        # ── 3. Load-balance bias (non-gradient buffer, grows with blocks) ─
        # Starts as a scalar 0.0; expanded in _ensure_bias() as chain grows.
        self.register_buffer("lb_bias", torch.zeros(1))

        # ── 4. Confidence MLP ──────────────────────────────────────────
        # Input: scalar normalised entropy H̃ ∈ [0, 1]
        # Output: confidence score c ∈ (0, 1)
        # A single learned affine + sigmoid so the model can shift the
        # confidence threshold rather than relying on raw entropy alone.
        self.conf_head = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    # ── helpers ───────────────────────────────────────────────────────

    def _ensure_bias(self, n: int) -> None:
        """Grow load-balance bias buffer to length n if needed."""
        if self.lb_bias.shape[0] < n:
            extra = torch.zeros(
                n - self.lb_bias.shape[0],
                device=self.lb_bias.device,
                dtype=self.lb_bias.dtype,
            )
            self.lb_bias = torch.cat([self.lb_bias, extra])

    @staticmethod
    def _mean_pool(
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """(B, S, D) → (B, D) respecting padding."""
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        return h.mean(1)

    def _update_load_bias(
        self,
        routing_weights: torch.Tensor,    # (B, n)
        n: int,
    ) -> None:
        """Online DeepSeek-style bias update (no grad, in-place buffer).

        f_i = mean fraction of routing weight assigned to block i across
              the batch.  Target = 1/n (uniform load).
        """
        if not self.training:
            return
        with torch.no_grad():
            f = routing_weights.mean(dim=0)          # (n,)
            target = 1.0 / n
            delta = self.bias_lr * (target - f)      # positive if under-used
            self.lb_bias[:n] = self.lb_bias[:n] + delta

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        block_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        embedding_hidden: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Aggregate block outputs via UCLBR.

        Parameters
        ----------
        block_outputs : list of (B, S, D)
            One tensor per block, frozen blocks first.
        attention_mask : (B, S) optional
        embedding_hidden : (B, S, D) optional
            Original token embeddings used as Q (same anchor as S-QKV).
            Falls back to the last block's output when not provided
            (degrades gracefully, e.g. during unit tests).

        Returns
        -------
        (B, S, D)
        """
        n = len(block_outputs)
        if n == 1:
            return block_outputs[0]

        self._ensure_bias(n)
        B, S, D = block_outputs[0].shape
        h_dim, d = self.n_heads, self.head_dim

        # Use last block output as Q fallback if no embedding provided
        query_src = embedding_hidden if embedding_hidden is not None else block_outputs[-1]

        # ── Step 1: Read-ME pre-gate ───────────────────────────────────
        # Compute relevance score r_i for each block from its mean-pool.
        # Shape: (B, n)
        pooled = torch.stack(
            [self._mean_pool(h, attention_mask) for h in block_outputs], dim=1
        )  # (B, n, D)
        relevance = self.pre_gate(pooled).squeeze(-1)   # (B, n)  ∈ (0,1)

        if self.top_k > 0 and self.top_k < n:
            # Hard sparsity: zero out all but top-k by relevance
            topk_vals, topk_idx = relevance.topk(self.top_k, dim=-1)
            mask_hard = torch.zeros_like(relevance)
            mask_hard.scatter_(1, topk_idx, 1.0)
            relevance = relevance * mask_hard

        # ── Step 2: Routing attention with load-balance bias ───────────
        # Q from frozen embeddings (no projection), shape (B, h, S, d)
        Q = query_src.view(B, S, h_dim, d).transpose(1, 2)     # (B, h, S, d)

        # K, V from all blocks concatenated along sequence axis
        blocks_cat = torch.cat(block_outputs, dim=1)            # (B, n*S, D)
        K = self.k_proj(blocks_cat).view(B, n * S, h_dim, d).transpose(1, 2)
        V = self.v_proj(blocks_cat).view(B, n * S, h_dim, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, h, S, n*S)

        # Mask padding keys
        if attention_mask is not None:
            kv_mask = attention_mask.unsqueeze(1).repeat(1, n, 1)   # (B, n*S)
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)             # (B,1,1,n*S)
            scores = scores.masked_fill(kv_mask == 0, float("-inf"))

        # Sequence-level routing weights by averaging attention across positions
        # and heads: (B, h, S, n*S) → (B, n)
        scores_seq = scores.mean(dim=(1, 2))            # (B, n*S)  avg over heads+positions
        scores_by_block = scores_seq.view(B, n, S).mean(dim=-1)   # (B, n) avg over tokens

        # Add load-balance bias (sequence-level only, not token-level)
        biased_logits = scores_by_block + self.lb_bias[:n].unsqueeze(0)  # (B, n)

        # Apply pre-gate: multiply logits by relevance before softmax
        # Blocks with low relevance get their logits suppressed
        biased_logits = biased_logits * relevance

        routing_weights = F.softmax(biased_logits, dim=-1)           # (B, n)

        # Update load-balance bias online
        self._update_load_bias(routing_weights.detach(), n)

        # ── Step 3: Uncertainty-calibrated confidence ──────────────────
        # Normalised entropy H̃ = H / log(n)  ∈ [0, 1]
        # c = learned function of H̃; interpolate router ↔ uniform
        eps = 1e-9
        H = -(routing_weights * (routing_weights + eps).log()).sum(dim=-1, keepdim=True)  # (B,1)
        H_norm = H / math.log(n)                                     # (B,1) ∈ [0,1]
        c = self.conf_head(H_norm)                                   # (B,1) ∈ (0,1)

        uniform = torch.full_like(routing_weights, 1.0 / n)          # (B, n)
        final_w = c * routing_weights + (1.0 - c) * uniform          # (B, n)

        # ── Step 4: Token-level weighted aggregation ───────────────────
        # Full token-level attention output using final routing weights
        attn_out = torch.matmul(
            F.softmax(scores, dim=-1),  # (B, h, S, n*S)
            V,                          # (B, h, n*S, d)
        )  # (B, h, S, d)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)         # (B, S, D)

        # Scale token-level output by per-block confidence-adjusted weights
        # Reshape final_w to (B, 1, n) so we can broadcast over sequence
        # Compute per-position weighted sum across blocks
        block_stack = torch.stack(block_outputs, dim=2)              # (B, S, n, D)
        w_expand = final_w.unsqueeze(1).unsqueeze(-1)                # (B, 1, n, 1)
        weighted_blocks = (block_stack * w_expand).sum(dim=2)        # (B, S, D)

        # Blend: 50% pure block aggregation + 50% attention-refined output
        # This keeps the routing interpretable while leveraging attention quality
        out = 0.5 * weighted_blocks + 0.5 * attn_out

        return self.out_proj(out)                                     # (B, S, D)
