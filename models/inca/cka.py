"""INCA CKA reference-set monitor  (Phase 1, T1.5).

Caches a fixed reference set at the start of each period and computes
linear CKA between the current block's encoder hidden states and the
reference every K_eval optimiser steps.

Linear CKA (Kornblith et al., ICML 2019):
    CKA(H1, H2) = HSIC_c(H1 H1ᵀ, H2 H2ᵀ)
                  ─────────────────────────────────────────────────────
                  sqrt( HSIC_c(H1 H1ᵀ, H1 H1ᵀ) · HSIC_c(H2 H2ᵀ, H2 H2ᵀ) )

where HSIC_c uses the unbiased centred-kernel estimator.

A CKA value near 1.0 means the representations have not changed since
the reference was cached → the block is representationally stable →
contributes to the block-full signal in inca_plateau.py.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ── Maths ────────────────────────────────────────────────────────────────────

def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012).

    K, L : (n, n) centred Gram matrices.
    Returns scalar.
    """
    n = K.shape[0]
    if n < 4:
        return K.new_zeros(1).squeeze()
    # Zero diagonal
    K = K - torch.diag(torch.diag(K))
    L = L - torch.diag(torch.diag(L))
    ones = torch.ones(n, 1, device=K.device, dtype=K.dtype)
    Krow = K @ ones          # (n, 1)
    Lrow = L @ ones          # (n, 1)
    term1 = (K * L).sum()
    term2 = (Krow.T @ Lrow).squeeze() * (2.0 / (n - 2))
    k_sum = K.sum()
    l_sum = L.sum()
    term3 = k_sum * l_sum / ((n - 1) * (n - 2))
    denom = n * (n - 3)
    return (term1 - term2 + term3) / denom


def linear_cka(H1: torch.Tensor, H2: torch.Tensor) -> float:
    """Compute linear CKA between two (n, d) representation matrices.

    Both matrices are mean-centred per feature before computing Gram
    matrices, which stabilises the estimator on small batches.
    Returns a float in [0, 1].  1.0 = identical representations.
    """
    with torch.no_grad():
        H1 = H1.float()
        H2 = H2.float()
        H1 = H1 - H1.mean(0, keepdim=True)
        H2 = H2 - H2.mean(0, keepdim=True)
        K = H1 @ H1.T
        L = H2 @ H2.T
        hsic_kl = _hsic_unbiased(K, L)
        hsic_kk = _hsic_unbiased(K, K)
        hsic_ll = _hsic_unbiased(L, L)
        denom = (hsic_kk * hsic_ll).clamp(min=0).sqrt()
        if denom < 1e-10:
            return 1.0
        return (hsic_kl / denom).clamp(0.0, 1.0).item()


# ── Monitor ──────────────────────────────────────────────────────────────────

class CKAMonitor:
    """Caches a fixed reference set and tracks CKA drift over training.

    Typical usage inside train_inca_v3.py::

        monitor = CKAMonitor(ref_size=cfg.cka_ref_size)

        # at the start of each period:
        monitor.cache_reference(model, period.probes, tokenizer, device, cfg)

        # every k_eval optimiser steps:
        cka = monitor.compute(model, device)
        if monitor.is_saturated(cfg.cka_saturation_threshold):
            ...  # feed into consensus detector
    """

    def __init__(self, ref_size: int = 200) -> None:
        self.ref_size = ref_size
        self._ref_hidden: Optional[torch.Tensor] = None  # (n, d) on CPU
        self._ref_enc_input: Optional[Dict[str, torch.Tensor]] = None
        self.history: List[float] = []

    # ── public API ───────────────────────────────────────────────────

    @torch.no_grad()
    def cache_reference(
        self,
        encoder: nn.Module,
        items: List[dict],
        tokenizer,
        device: str,
        max_seq_len: int = 256,
    ) -> None:
        """Cache hidden states from *encoder* on a random sample of *items*.

        Call once at the start of each period (before any training steps).
        """
        sample = random.sample(items, min(self.ref_size, len(items)))
        texts = self._items_to_texts(sample)
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding=True,
            return_tensors="pt",
        )
        self._ref_enc_input = {k: v.to(device) for k, v in enc.items()}

        encoder.eval()
        hidden = self._encode(encoder, self._ref_enc_input)
        self._ref_hidden = hidden.cpu()
        self.history = []

    @torch.no_grad()
    def compute(self, encoder: nn.Module, device: str) -> float:
        """Compute CKA between *encoder*'s current output and the reference.

        Returns 1.0 if the reference has not been cached yet (treat as
        maximally stable so it never falsely triggers saturation).
        """
        if self._ref_enc_input is None or self._ref_hidden is None:
            return 1.0

        encoder.eval()
        inp = {k: v.to(device) for k, v in self._ref_enc_input.items()}
        current = self._encode(encoder, inp).cpu()
        cka = linear_cka(self._ref_hidden, current)
        self.history.append(cka)
        return cka

    def is_saturated(self, threshold: float = 0.95) -> bool:
        """True if the latest CKA value ≥ *threshold* (representation stable)."""
        return bool(self.history) and self.history[-1] >= threshold

    def reset(self) -> None:
        """Call when a new block is created so history restarts."""
        self._ref_hidden = None
        self._ref_enc_input = None
        self.history = []

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _items_to_texts(items: List[dict]) -> List[str]:
        """Extract input text from heterogeneous item schemas.

        Supports (in priority order):
          * CC-News v2 stream:      "input_text"  (preferred)
          * Raw CC-News articles:   "text"
          * Legacy QA probes:       "question" + optional "evidence"
        """
        texts: List[str] = []
        for it in items:
            inp = (it.get("input_text") or "").strip()
            if inp:
                texts.append(inp)
                continue
            raw = (it.get("text") or "").strip()
            if raw:
                texts.append(raw[:400])
                continue
            q = (it.get("question") or "").strip()
            e = (it.get("evidence") or "").strip()
            texts.append(f"question: {q} context: {e[:400]}" if e else f"question: {q}")
        return texts

    @staticmethod
    def _encode(encoder: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run encoder and mean-pool over non-padding positions → (n, d).

        Handles two output types:
          * HuggingFace ModelOutput (has ``.last_hidden_state``) — standard T5 encoder
          * Raw tensor (B, S, D) — returned by INCALayerManager.forward()
        """
        out = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        # INCALayerManager returns a plain tensor; HF models return a ModelOutput
        hidden = out if isinstance(out, torch.Tensor) else out.last_hidden_state
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)
        return pooled.detach()
