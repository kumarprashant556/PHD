"""INCA block-chain layer manager  (Phase 1, T1.5 / Architecture).

Wraps a FLAN-T5 encoder and manages the growing chain of frozen + one
trainable INCA block.

Design
------
A single FLAN-T5 encoder has ``n_layers`` transformer layers.  INCA
partitions those layers into contiguous *blocks* of ``layers_per_block``
layers each.  When a block saturates:

  1. The current (trainable) block is frozen in place.
  2. Its weights are deep-copied to form a new trainable block (warm-start).
  3. A CrossAttentionSelector weight is added for the new block.
  4. The model grows until ``n_max_blocks`` is reached (then RuntimeError).

Forward pass
------------
Every block receives the same encoder input (the embedding + position
encoding from the *base* encoder).  Their hidden-state outputs are
aggregated by the CrossAttentionSelector into one (B, S, D) tensor that
the T5 decoder attends to.

The base model's own ``forward`` is not called end-to-end; instead:

  * Token + position embeddings are extracted via ``_embed()``.
  * Each block runs its layers sequentially over those embeddings.
  * All blocks are called even for inference so the selector can weight
    them correctly.
  * Frozen blocks run under ``torch.no_grad()`` for efficiency.

Param groups
------------
``trainable_params()`` returns only the current (last) block's parameters
plus the selector gate, making it trivial to pass to an optimiser without
accidentally unfreezing history.

``grad_norm()`` computes the L2 norm of those same trainable parameters'
gradients (used by GradNormTracker in inca_plateau.py).
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .inca_selectors import CrossAttentionSelector, EmbeddingQuerySelector, WeightedSumSelector
from .inca_uclbr    import UCLBRSelector

# All selector variants available from day one for E-ROUTE ablation.
# Switch via cfg.selector in YAML.
_SELECTOR_CLS = {
    "embedding_query": EmbeddingQuerySelector,   # S-QKV  — recommended default
    "uclbr":           UCLBRSelector,            # UCLBR  — full three-component router
    "cross_attention": CrossAttentionSelector,   # S-FULL — MLP-gate sequence-level
    "weighted_sum":    WeightedSumSelector,      # S-WS   — blind scalar control
}
_DEFAULT_SELECTOR = "embedding_query"


# ── helpers ───────────────────────────────────────────────────────────────────

def _freeze(module: nn.Module) -> None:
    """Set all parameters of *module* to requires_grad=False."""
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module) -> None:
    """Set all parameters of *module* to requires_grad=True."""
    for p in module.parameters():
        p.requires_grad_(True)


# ── block wrapper ─────────────────────────────────────────────────────────────

class INCАBlock(nn.Module):
    """A slice of encoder transformer layers treated as a single INCA block.

    Parameters
    ----------
    layers : nn.ModuleList
        The ``layers_per_block`` encoder transformer layers belonging to
        this block.  Cloned from the base model; this module *owns* them.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run all layers in sequence.  Returns final hidden states (B, S, D)."""
        seq_len = hidden_states.shape[1]

        extended_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # T5-style: 0 → attend, large negative → mask
            extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9

        # Newer transformers T5Attention requires cache_position to derive
        # real_seq_length; pass a plain arange so it never falls back to None.
        cache_position = torch.arange(seq_len, device=hidden_states.device)

        for layer in self.layers:
            layer_out = layer(
                hidden_states,
                attention_mask=extended_mask,
                cache_position=cache_position,
            )
            # T5 layers return a tuple; first element is the new hidden state
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return hidden_states


# ── main manager ─────────────────────────────────────────────────────────────

class INCALayerManager(nn.Module):
    """Growing block-chain encoder manager.

    Parameters
    ----------
    base_model : transformers T5ForConditionalGeneration
        The full T5 model.  The manager extracts the encoder's embedding
        layer and transformer blocks; the decoder is left untouched.
    cfg : INCAConfig
        INCA configuration (layers_per_block, n_max_blocks, selector_hidden).
    """

    def __init__(self, base_model, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        self._base_model = base_model          # kept for decoder + lm-head
        encoder = base_model.encoder           # T5Stack

        # ── embedding layers (shared, frozen after init) ──────────────
        self.embed_tokens = encoder.embed_tokens
        # T5 has a final layer-norm on the encoder stack
        self.final_layer_norm = encoder.final_layer_norm

        # ── slice encoder layers into the first block ──────────────────
        all_layers = list(encoder.block)        # list[T5Block]
        lim = cfg.layers_per_block
        first_block_layers = nn.ModuleList(copy.deepcopy(all_layers[:lim]))
        first_block = INCАBlock(first_block_layers)

        self.blocks: nn.ModuleList = nn.ModuleList([first_block])

        # ── selector ──────────────────────────────────────────────────
        # Chosen via cfg.selector (default: "embedding_query").
        # "embedding_query" — EmbeddingQuerySelector (S-QKV):
        #     Q = frozen original embeddings; K,V from block outputs.
        #     Token-level selection; blocks compete to best reconstruct
        #     the original input meaning.
        # "cross_attention" — CrossAttentionSelector (S-FULL):
        #     MLP gate on mean-pooled block outputs; sequence-level.
        # "weighted_sum"    — WeightedSumSelector (S-WS):
        #     Input-independent scalar weight per block; control ablation.
        hidden_size = base_model.config.d_model
        self.d_model = hidden_size
        selector_name = getattr(cfg, "selector", _DEFAULT_SELECTOR)
        sel_cls = _SELECTOR_CLS.get(selector_name)
        if sel_cls is None:
            raise ValueError(
                f"Unknown selector '{selector_name}'. "
                f"Choose from: {list(_SELECTOR_CLS)}"
            )
        if selector_name == "embedding_query":
            self.selector = EmbeddingQuerySelector(
                hidden_size=hidden_size,
                n_heads=getattr(cfg, "selector_heads", 4),
            )
        elif selector_name == "uclbr":
            self.selector = UCLBRSelector(
                hidden_size=hidden_size,
                pre_gate_hidden=getattr(cfg, "uclbr_pre_gate_hidden", 64),
                n_heads=getattr(cfg, "uclbr_heads", 4),
                bias_lr=getattr(cfg, "uclbr_bias_lr", 1e-3),
                top_k=getattr(cfg, "uclbr_top_k", 0),
            )
        elif selector_name == "cross_attention":
            self.selector = CrossAttentionSelector(
                hidden_size=hidden_size,
                gate_hidden=cfg.selector_hidden,
            )
        else:  # weighted_sum
            self.selector = WeightedSumSelector(n_blocks_init=1)

        # ── inter-block projections (architecture (a) from memorandum) ─
        # One projection per transition: proj[i] maps Block-i output →
        # Block-(i+1) input.  Identity-initialised so the architecture is
        # function-preserving at grow time.  Added/frozen together with
        # the corresponding source block.
        # At init only one block exists so no projections yet.
        self.inter_block_projs: nn.ModuleList = nn.ModuleList()

        # ── dropout from base encoder (reuse rate) ─────────────────────
        dropout_rate = getattr(base_model.config, "dropout_rate", 0.1)
        self.dropout = nn.Dropout(p=dropout_rate)

        # ── freeze everything except the current (last) block ──────────
        _freeze(self.embed_tokens)
        _freeze(self.final_layer_norm)
        # First block starts trainable; selector always trainable

    # ── properties ────────────────────────────────────────────────────

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def current_block(self) -> INCАBlock:
        return self.blocks[-1]  # type: ignore[return-value]

    # ── grow ──────────────────────────────────────────────────────────

    def freeze_and_grow(self) -> None:
        """Freeze the current block and warm-start a new one.

        Raises
        ------
        RuntimeError
            If ``n_max_blocks`` would be exceeded.
        """
        if self.n_blocks >= self.cfg.n_max_blocks:
            raise RuntimeError(
                f"INCA: n_max_blocks={self.cfg.n_max_blocks} reached — "
                "cannot grow further.  Increase n_max_blocks or stop training."
            )

        # 1. Freeze current block
        _freeze(self.current_block)

        # 2. Add identity-initialised inter-block projection for this
        #    transition and freeze it together with the source block.
        #    Identity init keeps the architecture function-preserving at
        #    grow time: Block(i+1) initially sees the same activations it
        #    would have seen without the projection.
        proj = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.eye_(proj.weight)
        _freeze(proj)                          # frozen: source block is now frozen
        self.inter_block_projs.append(proj)

        # 3. Deep-copy to warm-start the new trainable block
        new_block = copy.deepcopy(self.current_block)
        _unfreeze(new_block)
        self.blocks.append(new_block)

        # 4. The selector auto-handles new number of blocks since it
        #    operates on the dynamic list; no structural change needed
        #    (CrossAttentionSelector's gate MLP re-uses the same weights
        #    for all blocks — each pooled repr is projected independently).

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input_ids through all blocks; return selector-aggregated (B, S, D).

        Frozen blocks run under ``torch.no_grad()`` for memory efficiency.
        The current (last) block always runs with gradients.
        """
        # Shared token embeddings (frozen)
        with torch.no_grad():
            hidden = self.embed_tokens(input_ids)
            hidden = self.dropout(hidden)

        block_outputs: List[torch.Tensor] = []
        # Architecture (a): sequential chain + original-embedding skip.
        # chain_hidden is threaded through blocks sequentially.
        # At each transition: chain_hidden = proj_i(chain_hidden) + embedding_hidden
        # so the original token embeddings act as a persistent residual skip
        # that prevents representation drift through deep frozen chains.
        embedding_hidden = hidden    # (B, S, D) — kept for skip connections
        chain_hidden = hidden        # starts as plain embeddings (Block 0 input)

        for i, block in enumerate(self.blocks):
            is_current = (i == len(self.blocks) - 1)

            # ── inter-block projection + embedding skip ────────────────
            # proj[i-1] maps Block-(i-1) output into Block-i's input space.
            # Adding embedding_hidden gives the block a direct path back to
            # the original token representations regardless of chain depth.
            if i > 0:
                proj = self.inter_block_projs[i - 1]
                if is_current:
                    chain_hidden = proj(chain_hidden) + embedding_hidden
                else:
                    with torch.no_grad():
                        chain_hidden = proj(chain_hidden) + embedding_hidden

            # ── block forward ──────────────────────────────────────────
            if is_current:
                h = block(chain_hidden, attention_mask=attention_mask)
                h = self.final_layer_norm(h)
                h = self.dropout(h)
            else:
                with torch.no_grad():
                    h = block(chain_hidden, attention_mask=attention_mask)
                    h = self.final_layer_norm(h)
                    h = self.dropout(h)

            chain_hidden = h          # pass to next block in chain
            block_outputs.append(h)

        # Aggregate via selector.
        # EmbeddingQuerySelector needs embedding_hidden as the fixed Q.
        # CrossAttentionSelector and WeightedSumSelector ignore it.
        combined = self.selector(
            block_outputs,
            attention_mask=attention_mask,
            embedding_hidden=embedding_hidden,   # original frozen embeddings
        )
        return combined  # (B, S, D)

    # ── parameter utilities ───────────────────────────────────────────

    def trainable_params(self) -> List[nn.Parameter]:
        """Return parameters that should be passed to the optimiser.

        Includes: current block + selector gate + incoming inter-block
        projection (if one exists).  The incoming projection was frozen
        together with its source block; we unfreeze it here so the current
        trainable block can learn how to use the frozen block's output.
        """
        params: List[nn.Parameter] = []
        params.extend(p for p in self.current_block.parameters() if p.requires_grad)
        params.extend(p for p in self.selector.parameters() if p.requires_grad)
        # The projection feeding into the current block (index n_blocks-2)
        # should be trainable so it can optimise the inter-block knowledge
        # transfer for the current period.
        if self.inter_block_projs:
            incoming_proj = self.inter_block_projs[-1]
            _unfreeze(incoming_proj)
            params.extend(p for p in incoming_proj.parameters() if p.requires_grad)
        return params

    @torch.no_grad()
    def grad_norm(self) -> float:
        """L2 norm of gradients over trainable parameters.

        Returns 0.0 if no gradients have been computed yet.
        """
        norms = [
            p.grad.detach().norm(2).item()
            for p in self.trainable_params()
            if p.grad is not None
        ]
        if not norms:
            return 0.0
        total = math.sqrt(sum(n ** 2 for n in norms))
        return total

    # ── state for checkpointing ───────────────────────────────────────

    def manager_state(self) -> dict:
        """Return a lightweight state dict for saving/restoring INCA growth state."""
        return {
            "n_blocks": self.n_blocks,
            "blocks_state": [b.state_dict() for b in self.blocks],
            "selector_state": self.selector.state_dict(),
            "proj_states": [p.state_dict() for p in self.inter_block_projs],
        }

    def load_manager_state(self, state: dict) -> None:
        """Restore block chain from a previously saved ``manager_state()``."""
        saved_n = state["n_blocks"]
        # Grow to match saved block count
        while self.n_blocks < saved_n:
            self.freeze_and_grow()
        # Load block weights
        for i, (block, sd) in enumerate(zip(self.blocks, state["blocks_state"])):
            block.load_state_dict(sd)
            if i < saved_n - 1:
                _freeze(block)
            else:
                _unfreeze(block)
        self.selector.load_state_dict(state["selector_state"])
        # Load projection weights (backwards-compatible: key may not exist in
        # old checkpoints that used the parallel architecture)
        for proj, sd in zip(self.inter_block_projs,
                            state.get("proj_states", [])):
            proj.load_state_dict(sd)
