"""Cloze evaluation (BWT/FWT scoring) and regret-matrix pretty-printer."""
from __future__ import annotations

import string
from typing import Any, Dict, List, Optional

import torch


@torch.no_grad()
def eval_cloze_accuracy(
    model,
    probes: List[Dict[str, Any]],
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_input_len: int = 256,
    max_new_tokens: int = 32,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    """Exact-match accuracy on entity_cloze + date_cloze probes.

    Optional ``autocast_dtype`` runs ``model.generate`` under ``torch.autocast``
    so eval matches the training precision (bf16/fp16) — same speed benefit as
    training, no extra fp32 conversion overhead.
    """
    cloze = [p for p in probes if p.get("probe_type") in ("entity_cloze", "date_cloze")]
    if not cloze:
        return 0.0

    def _norm(s: str) -> str:
        return " ".join(
            s.lower().translate(str.maketrans("", "", string.punctuation)).split()
        )

    # Resolve the model's actual device (Trainer may have moved it).
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device)

    # Map torch.device to autocast device_type string.
    device_type = model_device.type  # "cuda" | "mps" | "cpu"

    model.eval()
    correct = total = 0
    for i in range(0, len(cloze), batch_size):
        batch = cloze[i: i + batch_size]
        inputs = tokenizer(
            [p["input"] for p in batch],
            return_tensors="pt", padding=True, truncation=True, max_length=max_input_len,
        ).to(model_device)
        if autocast_dtype is not None and device_type in ("cuda", "mps"):
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        else:
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for pred, probe in zip(preds, batch):
            ans = probe.get("target") or probe.get("answer") or ""
            aliases = probe.get("aliases") or []
            accepted = {_norm(ans)} | {_norm(a) for a in aliases}
            if _norm(pred) in accepted:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def pretty_matrix(mat, period_ids: List[str]) -> str:
    """Format the BWT/FWT regret matrix as an aligned text table for the log.

    Rows = "after_period t"; columns = "probes of period j".
    The diagonal R[t, t] is in-period accuracy; below-diagonal = BWT contribution.
    """
    n = len(period_ids)
    col_w = max(8, max(len(p) for p in period_ids) + 2)
    head = "after_period".ljust(col_w) + "".join(p.center(col_w) for p in period_ids)
    rule = "─" * len(head)
    lines = [
        "Regret matrix R[t, j] = accuracy of model-after-t on probes-of-j",
        rule, head, rule,
    ]
    for t in range(n):
        row = [period_ids[t].ljust(col_w)]
        for j in range(n):
            row.append(f"{mat.get(t, j):.4f}".center(col_w))
        lines.append("".join(row))
    lines.append(rule)
    return "\n".join(lines)
