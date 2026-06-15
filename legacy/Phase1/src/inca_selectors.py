"""INCA routing / selector modules.

Three selectors available for ablation E-ROUTE:

  S-QKV  — EmbeddingQuerySelector  (recommended default)
  S-FULL — CrossAttentionSelector  (MLP-gate baseline)
  S-WS   — WeightedSumSelector     (blind scalar control)

EmbeddingQuerySelector  (S-QKV)
--------------------------------
Original idea: use the frozen token embeddings as a fixed Query and let
each block's output compete as Key/Value.

  Q  = original embedding  (B, S, D)  — fixed, no gradient, no projection
  K_i = W_k · block_i_output          — learned per-block key
  V_i = W_v · block_i_output          — learned per-block value

For each token position s the attention score against block i is:

    score(s, i) = Q[s] · K_i[s]^T / sqrt(d_k)

Softmax over the n_blocks dimension gives a per-token, per-position
mixture weight.  The final output is the weighted sum of all V_i[s].

Why fix Q to the original embedding?
  The embedding represents what the input token *is*.  We want to ask:
  "given what this token originally meant, which block's transformation
   of it is most useful?"  Keeping Q fixed means the question never
   drifts — only the blocks compete to be the best answer.

Compared with CrossAttentionSelector:
  - Token-level selection (not sequence-level); position 5 and position 50
    can independently weight different blocks.
  - No mean-pooling information loss.
  - Fewer parameters: only W_k and W_v; no Q projection.

CrossAttentionSelector  (S-FULL)
---------------------------------
Sequence-level MLP gate.  Mean-pools each block → scalar logit → softmax.
Same scalar weight applied to every token position.

WeightedSumSelector  (S-WS)
-----------------------------
Blind scalar weight per block, input-independent.  Pure control ablation.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingQuerySelector(nn.Module):
    """Fixed-query cross-attention selector  (ablation S-QKV).

    Q  = original token embeddings  — frozen, no projection, no gradient.
    K  = learned linear projection of each block's output.
    V  = learned linear projection of each block's output.

    For every token position the attention score measures how well block i's
    output matches the original embedding at that position.  Softmax over
    the block dimension gives a per-token mixture; the output is the
    weighted sum of all blocks' value projections.

    Parameters
    ----------
    hidden_size : int
        Encoder hidden dimension D.
    n_heads : int
        Number of attention heads.  Must divide hidden_size evenly.
        Default 4 — deliberately small to keep selector lightweight.
    dropout : float
        Attention dropout.  0.0 in eval mode automatically.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}"
        )
        self.hidden_size = hidden_size
        self.n_heads     = n_heads
        self.head_dim    = hidden_size // n_heads
        self.scale       = self.head_dim ** -0.5

        # No Q projection — raw embeddings are the query.
        # K and V share a single projection matrix (tied weights).
        # Tying K and V keeps param count minimal and works well
        # when blocks all live in the same representation space.
        # Set tied=False to decouple if ablation warrants it.
        self.k_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)

        # Identity-initialise projections so the selector is
        # function-preserving at grow time (output ≈ mean of block outputs).
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

    def forward(
        self,
        block_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        embedding_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate block outputs using fixed embedding queries.

        Parameters
        ----------
        block_outputs : list of (B, S, D)
            One tensor per block, frozen blocks first.
        attention_mask : (B, S) bool/int optional
            1 = real token, 0 = padding.  Prevents padding tokens from
            contaminating attention scores.
        embedding_hidden : (B, S, D) optional
            Raw token embeddings (before any block).  Must be provided
            when n_blocks > 1; ignored (short-circuits) when n_blocks == 1.

        Returns
        -------
        (B, S, D)
        """
        if len(block_outputs) == 1:
            # Single block: no aggregation needed.
            return block_outputs[0]

        if embedding_hidden is None:
            raise ValueError(
                "EmbeddingQuerySelector requires embedding_hidden when "
                "n_blocks > 1.  Pass it from INCALayerManager.forward()."
            )

        B, S, D = embedding_hidden.shape
        n = len(block_outputs)
        h, d = self.n_heads, self.head_dim

        # ── Query: raw embeddings, reshaped for multi-head ──────────────
        # No projection — embeddings are used as-is.
        # Shape: (B, h, S, d)
        Q = embedding_hidden.view(B, S, h, d).transpose(1, 2)   # (B, h, S, d)

        # ── Keys and Values: one per block ──────────────────────────────
        # Stack all block outputs along the "block" axis first, project
        # once, then split back.  More efficient than n separate calls.
        #
        # blocks_cat : (B, n*S, D)
        blocks_cat = torch.cat(block_outputs, dim=1)

        K = self.k_proj(blocks_cat).view(B, n, S, h, d)   # (B, n, S, h, d)
        V = self.v_proj(blocks_cat).view(B, n, S, h, d)

        # Reshape to (B, h, n*S, d) so matmul gives (B, h, S, n*S)
        K = K.permute(0, 3, 1, 2, 4).reshape(B, h, n * S, d)  # (B, h, n*S, d)
        V = V.permute(0, 3, 1, 2, 4).reshape(B, h, n * S, d)

        # ── Scaled dot-product attention ────────────────────────────────
        # scores : (B, h, S, n*S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Mask padding positions in K so they don't attract attention.
        if attention_mask is not None:
            # attention_mask: (B, S)  →  (B, 1, 1, n*S)
            # Repeat the mask n times (once per block) along the key axis.
            kv_mask = attention_mask.unsqueeze(1).repeat(1, n, 1)   # (B, n*S)
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)             # (B,1,1,n*S)
            scores = scores.masked_fill(kv_mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)     # (B, h, S, n*S)
        # Replace any NaN from all-padding rows with uniform weights
        weights = torch.nan_to_num(weights, nan=1.0 / (n * S))
        weights = self.attn_drop(weights)

        # ── Weighted sum → output ────────────────────────────────────────
        # (B, h, S, n*S) × (B, h, n*S, d) → (B, h, S, d)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).reshape(B, S, D)   # (B, S, D)
        return self.out_proj(out)


class CrossAttentionSelector(nn.Module):
    """Softmax-gated aggregation of per-block encoder hidden states.

    Each block contributes a hidden-state tensor of shape (B, S, D).
    The selector computes a scalar gate logit per block by projecting the
    mean-pooled representation through a two-layer MLP, then takes a
    softmax over all blocks.  The weighted sum of block outputs is
    returned as the combined encoder representation.

    Parameters
    ----------
    hidden_size : int
        Encoder hidden dimension (D).
    gate_hidden : int
        Hidden units in the gate MLP.  Defaults to hidden_size // 4.
    """

    def __init__(self, hidden_size: int, gate_hidden: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **_kwargs,                                   # absorb embedding_hidden etc.
    ) -> torch.Tensor:
        """Aggregate block outputs into a single tensor.

        Parameters
        ----------
        hidden_states : list of (B, S, D)
            One tensor per block (frozen blocks first, current block last).
        attention_mask : (B, S) optional
            Used for mean-pooling.  If None, simple mean over sequence.

        Returns
        -------
        (B, S, D) — weighted combination of all block outputs.
        """
        if len(hidden_states) == 1:
            return hidden_states[0]

        # Mean-pool each block's output → (B, D)
        pooled = [self._mean_pool(h, attention_mask) for h in hidden_states]

        # Scalar gate logit per block → (B, n_blocks)
        logits = torch.cat(
            [self.gate(p) for p in pooled], dim=-1
        )  # (B, n_blocks)
        weights = F.softmax(logits, dim=-1)  # (B, n_blocks)

        # Weighted sum over sequence-level tensors → (B, S, D)
        stacked = torch.stack(hidden_states, dim=1)      # (B, n, S, D)
        w = weights.unsqueeze(-1).unsqueeze(-1)          # (B, n, 1, 1)
        out = (stacked * w).sum(dim=1)                   # (B, S, D)
        return out

    @staticmethod
    def _mean_pool(
        h: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean-pool (B, S, D) → (B, D) respecting padding."""
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        return h.mean(1)


class WeightedSumSelector(nn.Module):
    """Simpler learned scalar-per-block weighted sum (ablation S-WS).

    A single learnable scalar weight per block; softmax normalised.
    Weaker than CrossAttentionSelector but useful as a control.
    """

    def __init__(self, n_blocks_init: int = 1) -> None:
        super().__init__()
        # logits are grown dynamically; start with one weight
        self.logits = nn.Parameter(torch.zeros(n_blocks_init))

    def grow(self) -> None:
        """Add one new weight logit when a block is frozen and a new one added."""
        with torch.no_grad():
            new = torch.zeros(1, device=self.logits.device)
            self.logits = nn.Parameter(torch.cat([self.logits, new]))

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **_kwargs,                                   # absorb embedding_hidden etc.
    ) -> torch.Tensor:
        n = len(hidden_states)
        # Pad / trim logits if needed
        logits = self.logits[:n] if len(self.logits) >= n else F.pad(
            self.logits, (0, n - len(self.logits))
        )
        weights = F.softmax(logits, dim=0)           # (n,)
        stacked = torch.stack(hidden_states, dim=1)  # (B, n, S, D)
        w = weights.view(1, n, 1, 1)
        return (stacked * w).sum(1)                  # (B, S, D)
