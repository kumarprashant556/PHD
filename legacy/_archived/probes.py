"""Linear probing on frozen INCA blocks  (capsel/evaluation/probes.py)

Used by E-CLS2 ablation:
  For each frozen block, fit a linear probe on mean-pooled representations
  and measure accuracy on a held-out probe set.

Expected result: monotone accuracy decrease from Block 0 → Block N
(earlier blocks encode more general features; later = more specialised).
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def mean_pool(hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean-pool token representations, respecting attention mask."""
    if mask is None:
        return hidden.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)


class LinearProbe(nn.Module):
    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def probe_block(
    block,
    dataloader: DataLoader,
    n_classes: int,
    device: torch.device,
    n_epochs: int = 10,
    lr: float = 1e-3,
) -> float:
    """Train a linear probe on top of `block` and return validation accuracy.

    Parameters
    ----------
    block      : frozen INCАBlock (or any nn.Module returning (B, S, D))
    dataloader : yields (input_ids, attention_mask, labels) — labels are class ints
    n_classes  : number of probe classes
    """
    probe = LinearProbe(block.layers[0].layer[0].SelfAttention.q.weight.shape[1],
                        n_classes).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    block.eval()
    for _ in range(n_epochs):
        for batch in dataloader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            with torch.no_grad():
                h = block(ids, attention_mask=mask)
            pooled = mean_pool(h, mask)
            logits = probe(pooled)
            loss   = criterion(logits, lbls)
            optimiser.zero_grad(); loss.backward(); optimiser.step()

    # Evaluate
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            h      = block(ids, attention_mask=mask)
            pooled = mean_pool(h, mask)
            preds  = probe(pooled).argmax(dim=-1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    return correct / max(1, total)


def probe_all_blocks(
    layer_manager,
    dataloader: DataLoader,
    n_classes: int,
    device: torch.device,
) -> List[float]:
    """Return probe accuracy for each frozen block in the manager."""
    accs = []
    for i, block in enumerate(layer_manager.blocks[:-1]):  # skip current trainable
        acc = probe_block(block, dataloader, n_classes, device)
        accs.append(acc)
        print(f"  Block {i} probe accuracy: {acc:.3f}")
    return accs
