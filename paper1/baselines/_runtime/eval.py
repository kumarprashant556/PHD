"""Cloze evaluation (BWT/FWT scoring) and regret-matrix pretty-printer."""
from __future__ import annotations

import string
from collections import Counter
from typing import Any, Dict, List, Optional

import torch


def _token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1 between two strings.

    Tokenises on whitespace after lowercasing and stripping punctuation.
    Returns the harmonic mean of token precision and recall (bag-of-words).

    Why token F1 instead of exact match
    ------------------------------------
    Exact match is structurally zero for open-ended math / code tasks: the
    model rarely reproduces the target verbatim, even when it has learned the
    correct answer.  Token F1 awards partial credit for partially-correct
    outputs (e.g. "\\frac{3}{4}" vs "3/4" share "3" and "4") and gives full
    credit for correct short answers that appear anywhere in the generation
    ("The answer is 42" vs "42": recall=1, F1=high).

    Why not BERTScore
    -----------------
    BERTScore runs a full BERT forward pass per prediction-reference pair.
    ``_eval_accuracy`` is called every ``k_eval=50`` training steps, so
    BERTScore would add ~30 s × 27 calls ≈ 13 min of overhead per seed —
    unacceptable inside the training loop.  Token F1 runs in microseconds.
    """
    def _norm_tokens(s: str) -> List[str]:
        return s.lower().translate(str.maketrans("", "", string.punctuation)).split()

    pred_toks = _norm_tokens(pred)
    gold_toks = _norm_tokens(gold)
    if not pred_toks or not gold_toks:
        return 0.0

    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_counts.values())
    recall    = common / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall)


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


@torch.no_grad()
def eval_token_f1(
    model,
    items: List[Dict[str, Any]],
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_input_len: int = 256,
    max_new_tokens: int = 256,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    """Mean token-level F1 on input_text → target_text items.

    Used for domain_sequential (math / code / science) BWT/FWT scoring.
    Returns the mean of per-example SQuAD-style token F1 scores in [0, 1].

    Unlike ``eval_cloze_accuracy`` there is no probe_type filter — every item
    in ``items`` is scored.  Partial credit is awarded for partially-correct
    outputs, making the score a smooth signal suitable for both BWT/FWT
    reporting and INCA's saturation detector.

    max_new_tokens=256 is the default to accommodate full-completion targets
    (multi-step math solutions, complete Python programs) which can be 100-400
    tokens long.  Increase further if truncation is observed in generation logs.
    """
    if not items:
        return 0.0

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device)

    device_type = model_device.type
    model.eval()
    total_f1 = 0.0
    total     = 0
    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        inputs = tokenizer(
            [p["input_text"] for p in batch],
            return_tensors="pt", padding=True, truncation=True, max_length=max_input_len,
        ).to(model_device)
        if autocast_dtype is not None and device_type in ("cuda", "mps"):
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        else:
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for pred, probe in zip(preds, batch):
            ans = (probe.get("target_text") or probe.get("target")
                   or probe.get("answer") or "")
            total_f1 += _token_f1(pred, ans)
            total    += 1
    return total_f1 / total if total > 0 else 0.0


# Backwards-compatible alias: the cloze eval pipeline still imports this name.
eval_exact_match = eval_token_f1


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
