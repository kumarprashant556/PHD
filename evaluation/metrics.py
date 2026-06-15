"""CAPSEL-defined evaluation metrics (v2: seq2seq + causal).

Every metric here is specified in the CAPSEL Research Memorandum (FINAL,
Part XIII). This module is fully self-contained.

Primary metrics (v2 memo Part XV):

* ``BWT``  — backward transfer: Lopez-Paz & Ranzato 2017. For period p
  trained at step t and re-evaluated at the end of the stream (step T),
  ``BWT = mean_{p < T} ( acc(p, T) - acc(p, p) )``.
* ``ACC`` — average accuracy across all periods at end of stream:
  ``ACC = mean_p acc(p, T)``.
* ``FWT`` — forward transfer (zero-shot accuracy above chance before a
  period is trained): ``FWT = mean_p ( acc(p, p-1) - b_chance )``.
* ``RIR`` — relative improvement rate (CAPSEL XIII.4):
  ``RIR = (score_t - score_0) / max(score_0, chance)``, dataset-agnostic.

Seq2seq metrics (Track A — RealtimeQA / StreamingQA / TemporalWiki):

* ``exact_match``             — normalised string equality (SQuAD convention).
* ``token_f1``                — bag-of-words F1 on whitespace tokens.
* ``seq2seq_probe_accuracy``  — EM + F1 via model.generate(); returns
  ``(em_score, f1_score, rows)``.
* ``seq2seq_perplexity``      — mean per-token NLL of answer tokens given
  question+context as encoder input; ``exp(mean_loss)``.
* ``seq2seq_combined_score``  — ``em_weight*EM + f1_weight*F1``.

Causal-LM metrics (Track B — cc_news):

* ``CKA``                     — centred kernel alignment (CAPSEL XIII.2).
* ``fisher_convergence_rate`` — CAPSEL XIII.3. Ratio of L1 Fisher change.
* ``perplexity``              — mean causal-LM loss then ``exp``.
* ``probe_accuracy``          — MCQ accuracy via per-token perplexity.
* ``combined_score``          — 0.4×ppl_score + 0.6×probe_accuracy.

None of these metrics take model gradients; they are strictly read-only.
"""

from __future__ import annotations

import csv
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None                          # type: ignore[assignment]
    F = None                              # type: ignore[assignment]
    _HAS_TORCH = False

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):                  # type: ignore[misc]
        return it


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "This metric needs PyTorch installed. Run "
            "`pip install torch transformers --break-system-packages`."
        )


# ── Text normalisation helpers ────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> bool:
    """Normalised exact match (SQuAD / RealtimeQA convention)."""
    return _normalise(pred) == _normalise(gold)


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-words token F1 (SQuAD convention). Returns float in [0, 1]."""
    pred_toks = _normalise(pred).split()
    gold_toks = _normalise(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = sum(
        min(pred_toks.count(t), gold_toks.count(t))
        for t in set(pred_toks) & set(gold_toks)
    )
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall    = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ── Seq2seq probe accuracy (EM + F1 via generation) ──────────────────────────

def seq2seq_probe_accuracy(
    model,
    tokenizer,
    device: str,
    probes: Sequence[Dict[str, Any]],
    max_n: int = 200,
    max_new_tokens: int = 32,
    max_input_len: int = 256,
    record_csv: Optional[Path] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Generate answers for each probe and score with EM + F1.

    Probe schema (RealtimeQA / StreamingQA / TemporalWiki):
    ``{"question": str, "answer": str, "evidence": str | None,
       "choices": {"A": ..., ...}, "answer_key": "A",
       "date": str, "period": int}``

    Returns ``(em_score, f1_score, rows)``.
    """
    _require_torch()
    if not probes:
        return 0.0, 0.0, []

    model.eval()
    em_total  = 0
    f1_total  = 0.0
    seen      = 0
    rows: List[Dict[str, Any]] = []

    for item in _tqdm(probes[:max_n], desc="    eval probes", unit="probe",
                      leave=False, dynamic_ncols=True, position=2):
        question = item.get("question", "").strip()
        evidence = (item.get("evidence", "") or "").strip()

        # Resolve gold: prefer open "answer", fall back to MCQ key lookup
        gold = (item.get("answer", "") or "").strip()
        if not gold:
            key     = str(item.get("answer_key", ""))
            choices = item.get("choices", {})
            if isinstance(choices, dict):
                gold = choices.get(key, "")
            elif isinstance(choices, list) and key.isdigit():
                idx  = int(key)
                gold = choices[idx] if idx < len(choices) else ""
        gold = gold.strip()

        if not question or not gold:
            continue

        # Format encoder input (FLAN-T5 style)
        if evidence:
            input_text = f"question: {question} context: {evidence[:400]}"
        else:
            input_text = f"question: {question}"

        enc = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        em  = int(exact_match(pred, gold))
        f1v = token_f1(pred, gold)
        em_total += em
        f1_total += f1v
        seen     += 1
        rows.append({
            "question":  question[:120],
            "gold":      gold,
            "predicted": pred,
            "em":        em,
            "f1":        round(f1v, 4),
        })

    em_score = em_total / seen if seen else 0.0
    f1_score = f1_total / seen if seen else 0.0

    if record_csv is not None:
        record_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(record_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["question", "gold", "predicted", "em", "f1"])
            w.writeheader()
            w.writerows(rows)

    return em_score, f1_score, rows


def seq2seq_perplexity(
    model,
    probes: Sequence[Dict[str, Any]],
    tokenizer,
    device: str,
    max_n: int = 300,
    max_input_len: int = 256,
    max_answer_len: int = 32,
) -> float:
    """Mean per-token NLL of answer tokens given question+context encoder input.

    Uses teacher-forcing (labels = answer tokens). Returns ``exp(mean_loss)``
    or ``float('inf')`` if nothing scored.
    """
    _require_torch()
    model.eval()
    total_loss, n = 0.0, 0

    for item in _tqdm(probes[:max_n], desc="    eval seq2seq PPL", unit="probe",
                      leave=False, dynamic_ncols=True, position=2):
        question = item.get("question", "").strip()
        evidence = (item.get("evidence", "") or "").strip()
        gold     = (item.get("answer", "") or "").strip()
        if not gold:
            key     = str(item.get("answer_key", ""))
            choices = item.get("choices", {})
            if isinstance(choices, dict):
                gold = choices.get(key, "")
        gold = gold.strip()
        if not question or not gold:
            continue

        input_text = (f"question: {question} context: {evidence[:400]}"
                      if evidence else f"question: {question}")

        enc = tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=max_input_len,
        ).to(device)
        label_enc = tokenizer(
            gold, return_tensors="pt",
            truncation=True, max_length=max_answer_len,
        ).to(device)
        labels = label_enc["input_ids"]
        labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

        with torch.no_grad():
            out = model(**enc, labels=labels)
        if out.loss is not None and torch.isfinite(out.loss):
            total_loss += out.loss.item()
            n += 1

    return math.exp(total_loss / n) if n > 0 else float("inf")


def seq2seq_combined_score(
    em: float,
    f1: float,
    em_weight: float = 0.4,
    f1_weight: float = 0.6,
) -> float:
    """``em_weight * EM + f1_weight * F1``."""
    return em_weight * em + f1_weight * f1


# ── Causal-LM perplexity ──────────────────────────────────────────────────────

def perplexity(
    model,
    items: Sequence[Dict[str, Any]],
    tokenizer,
    device: str,
    max_seq_len: int = 128,
    max_n: int = 300,
) -> float:
    """Mean-token-loss perplexity on held-out items.

    Items are ``{"text": ...}`` dicts; items shorter than 10 characters are
    skipped. Returns ``exp(mean_loss)`` or ``float('inf')`` if nothing scored.
    """
    _require_torch()
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for item in _tqdm(items[:max_n], desc="    eval PPL", unit="doc",
                          leave=False, dynamic_ncols=True, position=2):
            text = item.get("text", "")
            if len(text) < 10:
                continue
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(device)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            out = model(input_ids=ids, labels=ids)
            if out.loss is not None and torch.isfinite(out.loss):
                total_loss += out.loss.item()
                n += 1
    if n == 0:
        return float("inf")
    return math.exp(total_loss / n)


def ppl_to_score(ppl: float, decay: float = 0.3) -> float:
    """Shape a raw perplexity into ``[0, 1]``."""
    if not math.isfinite(ppl):
        return 0.0
    return max(0.0, min(1.0, math.exp(-(max(ppl, 1.0) - 1.0) * decay)))


# ── Causal-LM MCQ probe accuracy ─────────────────────────────────────────────

def _score_choice(
    model,
    tokenizer,
    device: str,
    prompt: str,
    choice: str,
) -> float:
    """Per-token cross-entropy of ``choice`` given ``prompt`` (lower = better)."""
    _require_torch()
    full_text = f"{prompt.strip()} {choice.strip()}"
    enc_full   = tokenizer(full_text, return_tensors="pt").to(device)
    enc_choice = tokenizer(" " + choice.strip(), return_tensors="pt")

    input_ids  = enc_full.input_ids
    total_len  = input_ids.shape[1]
    choice_len = enc_choice.input_ids.shape[1]
    prompt_len = max(1, total_len - choice_len)

    labels = input_ids.clone()
    if prompt_len < labels.shape[1]:
        labels[:, :prompt_len] = -100
    else:
        labels[:, :-1] = -100

    with torch.no_grad():
        out = model(**enc_full, labels=labels)
    return float(out.loss.item())


def probe_accuracy(
    model,
    tokenizer,
    device: str,
    probes: Sequence[Dict[str, Any]],
    max_n: int = 200,
    record_csv: Optional[Path] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Multiple-choice accuracy via per-token perplexity (causal LM).

    Supports both new RealtimeQA-style probes (evidence field) and old mc4
    cloze format. Returns ``(accuracy, rows)``.
    """
    if not probes:
        return 0.0, []

    correct = 0
    seen    = 0
    rows: List[Dict[str, Any]] = []

    for item in _tqdm(probes[:max_n], desc="    eval probes", unit="probe",
                      leave=False, dynamic_ncols=True, position=2):
        question = item.get("question", "")
        evidence = item.get("evidence", None) or ""
        choices  = item.get("choices", {})
        if isinstance(choices, list):
            choices = {str(i): c for i, c in enumerate(choices)}

        gold = str(item.get("answer_key", ""))
        if not gold:
            gold_ans = (item.get("answer", "") or "").strip()
            for k, v in choices.items():
                if _normalise(str(v)) == _normalise(gold_ans):
                    gold = k
                    break

        if not choices or not gold:
            continue

        fmt = item.get("format", "")
        if evidence.strip():
            prompt = evidence.strip()
        elif fmt == "mc4" and "____" in question:
            prompt = question.split("____", 1)[0].strip() or question[:200].strip()
        else:
            prompt = question.strip()

        losses: List[Tuple[float, str]] = []
        for key in sorted(choices.keys()):
            losses.append((_score_choice(model, tokenizer, device, prompt, choices[key]), key))
        losses.sort(key=lambda x: x[0])
        predicted  = losses[0][1]
        is_correct = predicted == gold
        correct   += int(is_correct)
        seen      += 1
        rows.append({
            "question":  question[:120],
            "gold":      gold,
            "predicted": predicted,
            "correct":   is_correct,
        })

    acc = correct / seen if seen else 0.0
    if record_csv is not None:
        record_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(record_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["question", "gold", "predicted", "correct"])
            w.writeheader()
            w.writerows(rows)
    return acc, rows


def combined_score(
    ppl: float,
    probe_acc: float,
    ppl_weight: float = 0.4,
    probe_weight: float = 0.6,
    ppl_decay: float = 0.3,
) -> float:
    """Causal-LM combined score: ``ppl_weight*ppl_score + probe_weight*probe_acc``."""
    return ppl_weight * ppl_to_score(ppl, ppl_decay) + probe_weight * probe_acc


# ── BWT / ACC / FWT (Lopez-Paz & Ranzato 2017 conventions) ──────────────────

@dataclass
class StreamAccuracyMatrix:
    """Lower-triangular acc matrix: ``M[t][p]`` = accuracy on period ``p``
    measured right after training period ``t``. ``p <= t`` always.

    The scalar stored here is:
    - seq2seq mode: F1 score (``seq2seq_probe_accuracy`` return[1])
    - causal mode:  probe_acc (MCQ accuracy via perplexity)
    """
    matrix: List[List[float]]
    labels: List[str]

    def set(self, t: int, p: int, value: float) -> None:
        while len(self.matrix) <= t:
            self.matrix.append([])
        while len(self.matrix[t]) <= p:
            self.matrix[t].append(0.0)
        self.matrix[t][p] = value

    def get(self, t: int, p: int) -> float:
        if t >= len(self.matrix) or p >= len(self.matrix[t]):
            return 0.0
        return self.matrix[t][p]


def bwt(mat: StreamAccuracyMatrix) -> float:
    """Backward transfer: ``mean_{p<T} ( acc(T, p) - acc(p, p) )``."""
    T = len(mat.matrix) - 1
    if T <= 0:
        return 0.0
    vals = [mat.get(T, p) - mat.get(p, p) for p in range(T)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def acc(mat: StreamAccuracyMatrix) -> float:
    """Average accuracy after the full stream: ``mean_p acc(T, p)``."""
    T = len(mat.matrix) - 1
    if T < 0:
        return 0.0
    n = T + 1
    return float(sum(mat.get(T, p) for p in range(n)) / n)


def fwt(mat: StreamAccuracyMatrix, chance: float = 0.0) -> float:
    """Forward transfer: zero-shot F1/acc on period ``p`` before training.

    Note: chance is 0.0 for open-answer EM/F1 (seq2seq), 0.25 for 4-way MCQ.
    """
    if len(mat.matrix) < 2:
        return 0.0
    vals = []
    for p in range(1, len(mat.matrix)):
        if p < len(mat.matrix[p - 1]):
            vals.append(mat.matrix[p - 1][p] - chance)
    return float(sum(vals) / len(vals)) if vals else 0.0


# ── CAPSEL XIII.4: relative improvement rate ─────────────────────────────────

def rir(score_t: float, score_0: float, chance: float = 0.0) -> float:
    """``(score_t - score_0) / max(score_0, chance)``."""
    return (score_t - score_0) / max(score_0, chance, 1e-9)


# ── CAPSEL XIII.2: linear CKA ────────────────────────────────────────────────

def cka(H_A, H_B) -> float:
    """Linear centred kernel alignment between two representation matrices."""
    _require_torch()

    def center(H):
        return H - H.mean(dim=0, keepdim=True)

    A = center(H_A)
    B = center(H_B)
    num = (A.T @ B).norm("fro") ** 2
    den = (A.T @ A).norm("fro") * (B.T @ B).norm("fro") + 1e-9
    return float((num / den).item())


# ── CAPSEL XIII.3: Fisher convergence rate ────────────────────────────────────

def diagonal_fisher(model, data_loader, device: str, max_batches: int = 50):
    """Empirical diagonal Fisher over a labelled eval batch."""
    _require_torch()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    seen = 0
    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch.get("labels", input_ids.clone()).to(device)
        model.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        seen += 1
    if seen == 0:
        return fisher
    for n in fisher:
        fisher[n] /= seen
    model.zero_grad()
    return fisher


def fisher_convergence_rate(F_t, F_prev) -> float:
    """``||F_t - F_prev||_1 / ||F_prev||_1``."""
    num = 0.0
    den = 0.0
    for k in F_t.keys() & F_prev.keys():
        num += float((F_t[k] - F_prev[k]).abs().sum().item())
        den += float(F_prev[k].abs().sum().item())
    return num / (den + 1e-9)
