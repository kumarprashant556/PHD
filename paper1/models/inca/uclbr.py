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
        # Per-block scalar relevance from its mean-pool. (B, n)
        pooled = torch.stack(
            [self._mean_pool(h, attention_mask) for h in block_outputs], dim=1
        )  # (B, n, D)
        relevance = self.pre_gate(pooled).squeeze(-1)            # (B, n) ∈ (0,1)

        if self.top_k > 0 and self.top_k < n:
            _, topk_idx = relevance.topk(self.top_k, dim=-1)
            mask_hard = torch.zeros_like(relevance)
            mask_hard.scatter_(1, topk_idx, 1.0)
            relevance = relevance * mask_hard

        # ── Step 2: Per-block routing attention (independent per block) ─
        # Q from frozen embeddings (no projection).
        Q = query_src.view(B, S, h_dim, d).transpose(1, 2)        # (B, h, S, d)

        if attention_mask is not None:
            k_pad = attention_mask.unsqueeze(1).unsqueeze(2)      # (B,1,1,S)
        else:
            k_pad = None

        per_block_out:   List[torch.Tensor] = []
        per_block_score: List[torch.Tensor] = []                  # block relevance (B,)

        for out_i in block_outputs:
            K_i = self.k_proj(out_i).view(B, S, h_dim, d).transpose(1, 2)
            V_i = self.v_proj(out_i).view(B, S, h_dim, d).transpose(1, 2)
            scores_i = torch.matmul(Q, K_i.transpose(-2, -1)) * self.scale  # (B, h, S, S)
            if k_pad is not None:
                scores_i = scores_i.masked_fill(k_pad == 0, float("-inf"))
            w_i = F.softmax(scores_i, dim=-1)                     # (B, h, S, S)
            w_i = torch.nan_to_num(w_i, nan=1.0 / S)
            attn_i = torch.matmul(w_i, V_i)                       # (B, h, S, d)
            attn_i = attn_i.transpose(1, 2).reshape(B, S, D)      # (B, S, D)
            per_block_out.append(attn_i)

            diag_i = scores_i.diagonal(dim1=-2, dim2=-1)          # (B, h, S)
            if attention_mask is not None:
                am = attention_mask.unsqueeze(1).float()          # (B, 1, S)
                denom = am.sum(dim=-1).clamp(min=1)               # (B, 1)
                diag_i = (diag_i * am).sum(dim=-1) / denom        # (B, h)
            else:
                diag_i = diag_i.mean(dim=-1)
            per_block_score.append(diag_i.mean(dim=-1))           # (B,)

        # Per-block routing logits (B, n)
        block_scores = torch.stack(per_block_score, dim=-1)       # (B, n)

        # ── Step 3: DeepSeek auxiliary-loss-free load-balance bias ─────
        # Bias added at the block level, then pre-gate relevance gates each block.
        biased_logits = (block_scores + self.lb_bias[:n].unsqueeze(0)) * relevance  # (B, n)
        routing_weights = F.softmax(biased_logits, dim=-1)        # (B, n)

        # Update load-balance bias online
        self._update_load_bias(routing_weights.detach(), n)

        # ── Step 4: Uncertainty-calibrated confidence ──────────────────
        eps = 1e-9
        H = -(routing_weights * (routing_weights + eps).log()).sum(dim=-1, keepdim=True)
        H_norm = H / math.log(n)                                  # (B, 1) ∈ [0, 1]
        c = self.conf_head(H_norm)                                # (B, 1) ∈ (0, 1)

        uniform = torch.full_like(routing_weights, 1.0 / n)       # (B, n)
        final_w = c * routing_weights + (1.0 - c) * uniform       # (B, n)

        # ── Step 5: Weighted sum of per-block attended outputs ─────────
        stacked = torch.stack(per_block_out, dim=1)               # (B, n, S, D)
        w       = final_w.unsqueeze(-1).unsqueeze(-1)             # (B, n, 1, 1)
        out     = (stacked * w).sum(dim=1)                        # (B, S, D)
        return self.out_proj(out)
