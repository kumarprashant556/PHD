"""Domain-sequential dataset loader  (data/domain_sequential.py)

Paper B four-domain curriculum for direct comparison with LLaMA-Pro (Wu et al., 2024).
Training order: P1_metamath → P2_evol_code → P3_science → P4_medical

P1 and P2 use the SAME datasets as LLaMA-Pro's SFT stage, enabling a dataset-controlled
comparison: identical training data, different expansion strategy (fixed vs saturation-driven).
P3–P4 extend beyond LLaMA-Pro's math+code scope to demonstrate generalisation.

Domain       | HF source                             | Size      | Framing
-------------|---------------------------------------|-----------|----------------------------------------
P1_metamath  | meta-math/MetaMathQA                  | ~395 000  | "solve: " + question → full CoT solution
P2_evol_code | theblackcat102/evol-codealpaca-v1     | ~111 272  | "code: " + instruction → full program
P3_science   | allenai/sciq                          | ~13 700   | "answer: " + support + question → answer
P4_medical   | openlifescienceai/medmcqa             | ~182 000  | "answer: " + question + options → option text

LLaMA-Pro comparison note:
  LLaMA-Pro SFT used MetaMath (395k) + Evol-CodeAlpaca (111k) achieving GSM8K=43.59,
  HumanEval=44.51 pass@1.  INCA uses the same P1/P2 data with token-level F1 as a
  unified smooth metric (necessary for saturation detection) instead of binary pass@1
  / exact-match.  This makes our comparison dataset-controlled: any difference in
  final domain accuracy is attributable to expansion strategy, not data choice.

Retained alternative loaders (not in DEFAULT_PERIODS):
  P5_commonsense — tau/commonsense_qa, 10.9k commonsense MCQ (inactive)
  P1_trivia      — trivia_qa (rc.nocontext), 138k factual QA; smooth F1 backup
  P1_gsm8k       — openai/gsm8k, 7.4k CoT solutions; limited by size (≤7k/period)
  P1_math        — competition math; near-zero F1, kept for ablations only

Usage
-----
from data.domain_sequential import load_domain_sequential_periods
periods = load_domain_sequential_periods(n_per_period=8_000)
# {"P1_metamath": Dataset, "P2_evol_code": Dataset,
#  "P3_science": Dataset, "P4_medical": Dataset}

# Subset to 2 domains (e.g. quick test):
periods = load_domain_sequential_periods(
    periods=["P1_metamath", "P2_evol_code"], n_per_period=8_000
)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── HF import guard: prevent local datasets/ folder from shadowing the package ─
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_sp_backup = sys.path[:]
sys.path = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)]
from datasets import Dataset, concatenate_datasets, load_dataset  # noqa: E402
sys.path[:] = _sp_backup
del _sp_backup, _REPO_ROOT

from ._base import finalise, clean_text, subsample, drop_short, keep_columns, STANDARD_COLS

# ── Canonical period order ────────────────────────────────────────────────────

DEFAULT_PERIODS: List[str] = [
    "P1_metamath",   # meta-math/MetaMathQA — grade-school + competition math CoT (LLaMA-Pro SFT data)
    "P2_evol_code",  # theblackcat102/evol-codealpaca-v1 — instruction → Python (LLaMA-Pro SFT data)
    "P3_science",    # allenai/sciq — passage + question → correct answer
    "P4_medical",    # openlifescienceai/medmcqa — medical MCQ → correct option text
    # P5_commonsense removed — tau/commonsense_qa not used in this experiment
]


# ── Per-domain raw loaders ────────────────────────────────────────────────────

def _extract_boxed_answer(solution: str) -> str:
    """Extract the final \\boxed{...} answer from a competition math solution.

    Returns the contents of the last \\boxed{} in the solution string (handling
    nested braces), normalised to a plain string.  Falls back to the full
    solution if no boxed expression is found.

    Examples
    --------
    "...so the answer is \\boxed{108}."  →  "108"
    "...we get \\boxed{\\frac{3}{4}}."   →  "\\frac{3}{4}"
    """
    import re as _re
    # Scan for the LAST \boxed{ and extract matching-brace content.
    last_pos = solution.rfind(r'\boxed{')
    if last_pos == -1:
        last_pos = solution.rfind(r'\boxed {')  # rare spacing variant
    if last_pos == -1:
        return solution.strip()
    start = solution.index('{', last_pos)
    depth = 0
    for i in range(start, len(solution)):
        if solution[i] == '{':
            depth += 1
        elif solution[i] == '}':
            depth -= 1
            if depth == 0:
                return solution[start + 1: i].strip()
    return solution.strip()   # unclosed brace — fall back


def _load_math(n: int, seed: int) -> Dataset:
    """hendrycks/competition_math — problem → full solution framing.

    The MATH dataset has ~12 500 training examples across algebra,
    counting & probability, geometry, number theory, and pre-calculus.
    Each example has: problem (str) · solution (str) · level · type · subject.

    We use the FULL multi-step solution as the target so the model learns to
    generate complete reasoning chains ending with \\boxed{answer}.  Token-level
    F1 (SQuAD-style) is used as the metric, which gives partial credit for
    correct intermediate steps and the final answer even when the generation
    does not exactly reproduce the reference solution word-for-word.

    Note: previously used 'lighteval/MATH' (script-backed, deprecated by HF).
    'qwedsacf/competition_math' is the Parquet mirror — same 12 500 examples,
    same problem/solution/level/type columns, no trust_remote_code required.
    """
    print("  [domain_sequential] P1_math: loading qwedsacf/competition_math …")
    ds = load_dataset("qwedsacf/competition_math", split="train")

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        for prob, sol in zip(batch["problem"], batch["solution"]):
            inputs.append("solve: " + clean_text(str(prob or "")))
            # Full solution as target — model learns complete reasoning chains.
            targets.append(clean_text(str(sol or "")))
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  math: build input/target", remove_columns=ds.column_names)
    return ds


def _extract_function_name(code: str) -> str:
    """Extract the snake_case function name from the first `def` line.

    Returns the name as a plain string (e.g. "fibonacci", "bubble_sort").
    Falls back to "" so the example is later dropped by drop_short.
    """
    import re as _re
    m = _re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', code)
    return m.group(1).strip() if m else ""


def _load_trivia(n: int, seed: int) -> Dataset:
    """trivia_qa (rc.nocontext) — open-domain trivia → factual answer framing.

    TriviaQA (Joshi et al., 2017) contains 138 384 trivia questions with
    crowd-sourced and web-sourced evidence.  We use the ``rc.nocontext`` split
    (question only, no evidence documents) so the model must rely on its
    parametric knowledge — a clean test of continual factual learning.

    Framing:
        input_text  = "answer: " + question
        target_text = canonical answer string (typically 1–5 words: person names,
                      places, dates, short phrases)

    Why TriviaQA as P1 (replacing competition_math):
        - FLAN-T5-large was FLAN-pre-trained on TriviaQA-style tasks → baseline
          F1 ≈ 10–20% before fine-tuning, rising to 30–50% with fine-tuning.
          That non-trivial slope means RIR > rir_negligible → BLOCK_FULL fires
          instead of EXHAUSTED, producing meaningful EXP_T timing data.
        - Short multi-word targets (median ≈ 1.5 words) give smooth token-F1
          partial credit — e.g. "Sinclair" scores 0.67 for "Sinclair Lewis".
        - 138 K examples: no n_per_period size constraint.
    """
    import re as _re
    print("  [domain_sequential] P1_trivia: loading trivia_qa (rc.nocontext) …")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        for question, answer in zip(batch["question"], batch["answer"]):
            q   = clean_text(str(question or ""))
            # answer is a dict with keys: value, aliases, normalized_value, …
            if isinstance(answer, dict):
                tgt = clean_text(str(answer.get("value", "") or ""))
            else:
                tgt = clean_text(str(answer or ""))
            inputs.append("answer: " + q)
            targets.append(tgt)
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  trivia: build input/target", remove_columns=ds.column_names)
    return ds


def _load_gsm8k(n: int, seed: int) -> Dataset:
    """openai/gsm8k — grade-school math word problems → full chain-of-thought solution.

    GSM8K (Cobbe et al., 2021) has 7 473 training examples.  Each problem is a
    grade-school arithmetic word problem.  The reference solution is a SHORT
    step-by-step reasoning chain (median ≈ 47 words, P95 ≈ 102 words) ending
    with '#### <final_number>'.

    Target framing — FULL solution (not just the number):
        input_text  = "solve: " + question
        target_text = cleaned reasoning steps + "#### " + final_answer

    Example:
        Q: "Natalia sold clips to 48 friends in April and half as many in May.
            How many clips altogether?"
        A: "Natalia sold 48/2 = 24 clips in May.
            Natalia sold 48+24 = 72 clips altogether in April and May. #### 72"

    Why full solution (not just "72"):
        - Multi-word target → token-F1 gives partial credit for correct steps
          AND for the final answer, giving a smooth score gradient ≠ binary.
        - FLAN-T5-large was FLAN-trained on GSM8K → chain-of-thought style is
          familiar; expected fine-tune F1 ≈ 10–25% (vs. competition math 0–2%).
        - Median 47-word solutions fit comfortably within max_target_length=256.

    The <<48/2=24>> arithmetic annotations in the raw data are stripped to plain
    arithmetic (e.g. "48/2 = 24") for cleaner tokenisation and fairer F1.

    Caveat: only 7 473 training examples → use n_per_period ≤ 7 000.
    Can be activated by replacing P1_trivia in DEFAULT_PERIODS with P1_gsm8k
    and updating configs/paper_b.yaml accordingly.
    """
    import re as _re
    print("  [domain_sequential] P1_gsm8k: loading openai/gsm8k …")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    def _strip_annotations(text: str) -> str:
        """Remove <<48/2=24>> style intermediate-arithmetic annotations."""
        return _re.sub(r'<<[^>]*>>', '', text).strip()

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        for question, answer in zip(batch["question"], batch["answer"]):
            q   = clean_text(str(question or ""))
            # Keep full reasoning chain; strip <<...>> annotations for clean text
            tgt = _strip_annotations(str(answer or ""))
            inputs.append("solve: " + q)
            targets.append(tgt)
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  gsm8k: build input/target", remove_columns=ds.column_names)
    return ds


def _load_metamath(n: int, seed: int) -> Dataset:
    """meta-math/MetaMathQA — grade-school + competition math → full CoT solution.

    MetaMath (Yu et al., 2023) augments GSM8K and MATH competition problems via
    four strategies: MetaReasoning (rephrasing), FOBAR (fill-in-the-blank swap),
    Self-Verification (confirm answer), and Backward Reasoning (reason from answer).
    Total: ~395 000 examples.  No n_per_period size constraint (vs GSM8K's 7 473 cap).

    This is one of the two datasets used in LLaMA-Pro's SFT stage, enabling a
    dataset-controlled comparison: same training data, different expansion strategy.

    Target framing — FULL chain-of-thought solution:
        input_text  = "solve: " + query
        target_text = full reasoning chain (annotations stripped)

    Format note:
        GSM8K-sourced rows end with "#### <number>" and may contain <<48/2=24>>
        arithmetic annotations — these are stripped to plain arithmetic.
        MATH-sourced rows may end with "The answer is: X" or "\\boxed{X}" — kept as-is.
        In both cases the full multi-step reasoning is preserved for smooth token-F1.

    Why MetaMath over raw GSM8K:
        - 53× more examples (395k vs 7.4k) → no per-period cap, smoother saturation curves
        - Augmented reformulations → more diverse F1 gradient signal per problem
        - Same data as LLaMA-Pro SFT (GSM8K=43.59, HumanEval=44.51) → direct comparison
        - FLAN-T5-large FLAN-trained on GSM8K-style CoT → expected baseline F1 ≈ 5–15%
    """
    import re as _re
    print("  [domain_sequential] P1_metamath: loading meta-math/MetaMathQA …")
    ds = load_dataset("meta-math/MetaMathQA", split="train")

    def _strip_annotations(text: str) -> str:
        """Remove <<48/2=24>> style arithmetic annotations, keep plain arithmetic."""
        return _re.sub(r'<<[^>]*>>', '', text).strip()

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        for query, response in zip(batch["query"], batch["response"]):
            q   = clean_text(str(query    or ""))
            # Full CoT solution; strip <<>> annotations from GSM8K-style rows
            tgt = _strip_annotations(clean_text(str(response or "")))
            inputs.append("solve: " + q)
            targets.append(tgt)
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  metamath: build input/target", remove_columns=ds.column_names)
    return ds


def _load_evol_code(n: int, seed: int) -> Dataset:
    """theblackcat102/evol-codealpaca-v1 — instruction → full Python program.

    Evol-CodeAlpaca (Luo et al., 2023 / WizardCoder) applies WizardLM's Evol-Instruct
    pipeline to code: seed problems are evolved across increasing complexity levels
    (add constraints, deepen, broaden, add reasoning steps, complicate input).
    Total: ~111 272 Python instruction-code pairs.

    This is the second of the two datasets used in LLaMA-Pro's SFT stage, enabling a
    dataset-controlled comparison: same training data, different expansion strategy.
    LLaMA-Pro achieved HumanEval pass@1 = 44.51 with this data.

    Framing:
        input_text  = "code: " + instruction
        target_text = full Python implementation (``output`` column)

    Columns (verified): instruction, output.
    """
    print("  [domain_sequential] P2_evol_code: loading theblackcat102/evol-codealpaca-v1 …")
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        for instr, out in zip(batch["instruction"], batch["output"]):
            inputs.append("code: " + clean_text(str(instr or "")))
            targets.append(clean_text(str(out   or "")))
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  evol_code: build input/target", remove_columns=ds.column_names)
    return ds


def _load_code(n: int, seed: int) -> Dataset:
    """flytech/python-codes-25k — instruction → full Python program framing.

    bigcode/the-stack-smol is gated (requires BigCode licence agreement).
    flytech/python-codes-25k is an ungated 49 k-row instruction-following
    dataset: each example has a natural-language task description
    (``instruction``) and a Python implementation (``output``).

    We use the FULL-COMPLETION framing:
        input_text  = "code: " + instruction
        target_text = full Python program (``output``)

    Token-level F1 is used as the metric, which gives partial credit for
    correct function signatures, variable names, and logic structure even when
    the generated code does not exactly match the reference implementation.
    This is the natural completion task: given a description, write the code.
    """
    print("  [domain_sequential] P2_code: loading flytech/python-codes-25k …")
    ds = load_dataset("flytech/python-codes-25k", split="train")

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        for instr, out in zip(batch["instruction"], batch["output"]):
            inputs.append("code: " + clean_text(str(instr or "")))
            # Full Python program as target — model learns complete implementations.
            targets.append(clean_text(str(out or "")))
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  code: build input/target", remove_columns=ds.column_names)
    return ds


def _load_science(n: int, seed: int) -> Dataset:
    """allenai/sciq — natural question → answer framing.

    SciQ has 13 679 science questions, each with a free-text support paragraph
    explaining the concept, the question, and the correct answer.
    We use the natural structure:
        input_text  = "answer: " + support_passage + " " + question
        target_text = correct_answer
    This avoids the arbitrary 50/50 split and makes the learning signal clean:
    the model reads the passage and question, then generates the answer.
    """
    print("  [domain_sequential] P3_science: loading allenai/sciq …")
    splits = []
    for split_name in ("train", "validation", "test"):
        try:
            splits.append(load_dataset("allenai/sciq", split=split_name))
        except Exception:
            pass
    if not splits:
        raise RuntimeError(
            "Could not load allenai/sciq from HuggingFace.  "
            "Check connectivity or pre-download with: "
            "datasets.load_dataset('allenai/sciq')"
        )
    ds = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        n_items = len(batch["question"])
        for support, question, answer in zip(
            batch.get("support", [""] * n_items),
            batch["question"],
            batch.get("correct_answer", [""] * n_items),
        ):
            ctx = clean_text(str(support or ""))
            q   = clean_text(str(question or ""))
            inp = "answer: " + " ".join(p for p in [ctx, q] if p)
            tgt = clean_text(str(answer or ""))
            inputs.append(inp)
            targets.append(tgt)
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  science: build input/target", remove_columns=ds.column_names)
    return ds


def _load_medical(n: int, seed: int) -> Dataset:
    """openlifescienceai/medmcqa — medical MCQ → seq2seq framing.

    MedMCQA (Pal et al., 2022) contains ~182 000 Indian medical entrance-exam
    questions across 2 400+ health topics (anatomy, pharmacology, surgery …).

    Columns (verified): question, opa, opb, opc, opd, cop (int, 0-indexed),
                        choice_type, exp, subject_name, topic_name.

    Seq2seq framing:
        input_text  = "answer: <question> Options: A. <opa> B. <opb> C. <opc> D. <opd>"
        target_text = text of the correct option (just the option, no explanation)

    We keep targets SHORT (option text only) so the model learns to directly
    produce the correct answer token sequence.  Explanations are hundreds of
    tokens and would dominate the seq2seq loss without adding evaluation signal
    (F1 is measured against the correct option string).
    """
    print("  [domain_sequential] P4_medical: loading openlifescienceai/medmcqa …")
    ds = load_dataset("openlifescienceai/medmcqa", split="train")

    _opt_labels = ["A", "B", "C", "D"]
    _opt_keys   = ["opa", "opb", "opc", "opd"]

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        n_items = len(batch["question"])
        for i in range(n_items):
            q    = clean_text(str(batch["question"][i] or ""))
            opts = [clean_text(str(batch[k][i] or "")) for k in _opt_keys]
            cop  = int(batch["cop"][i] if batch["cop"][i] is not None else 0)

            # Guard against out-of-range cop values
            cop  = max(0, min(cop, len(opts) - 1))

            opt_str = " ".join(f"{l}. {t}" for l, t in zip(_opt_labels, opts))
            inp = "answer: " + q + " Options: " + opt_str

            # Target = correct option text only (short, clean seq2seq signal)
            tgt = opts[cop]

            inputs.append(inp)
            targets.append(tgt)
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  medical: build input/target", remove_columns=ds.column_names)
    return ds


def _load_commonsense(n: int, seed: int) -> Dataset:
    """tau/commonsense_qa — commonsense reasoning framing.

    CommonsenseQA (Talmor et al., 2019) contains ~12 000 questions requiring
    everyday commonsense knowledge about concepts and their relationships.
    Each item has: question, five labeled choices (A–E), answerKey (letter).

    Framing:
        input_text  = "answer: " + question
                      + " Options: A. " + choice_A + " B. " + choice_B + …
        target_text = text of the correct choice (looked up by answerKey)

    Note: train + validation are merged (test labels are withheld by the
    dataset authors).
    """
    print("  [domain_sequential] P5_commonsense: loading tau/commonsense_qa …")
    splits = []
    for split_name in ("train", "validation"):
        try:
            splits.append(load_dataset("tau/commonsense_qa", split=split_name))
        except Exception:
            pass
    if not splits:
        raise RuntimeError(
            "Could not load tau/commonsense_qa from HuggingFace. "
            "Check connectivity or try: datasets.load_dataset('tau/commonsense_qa')"
        )
    ds = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

    def _build_input_target(batch: dict) -> dict:
        inputs, targets = [], []
        n_items = len(batch["question"])
        for i in range(n_items):
            q       = clean_text(str(batch["question"][i] or ""))
            choices = batch["choices"][i]          # {"label": [...], "text": [...]}
            labels  = choices["label"]
            texts   = [clean_text(t) for t in choices["text"]]
            key     = str(batch["answerKey"][i] or "A").strip().upper()

            opt_str = " ".join(f"{l}. {t}" for l, t in zip(labels, texts))
            inp = "answer: " + q + " Options: " + opt_str

            try:
                idx = labels.index(key)
                tgt = texts[idx]
            except (ValueError, IndexError):
                tgt = texts[0]   # safe fallback; will be short enough to keep

            inputs.append(inp)
            targets.append(tgt)
        return {"input_text": inputs, "target_text": targets}

    ds = ds.map(_build_input_target, batched=True, batch_size=512,
                desc="  commonsense: build input/target", remove_columns=ds.column_names)
    return ds


# ── Loader dispatch ───────────────────────────────────────────────────────────

_DOMAIN_LOADERS = {
    # ── Active DEFAULT_PERIODS (4-domain curriculum) ──────────────────────────
    "P1_metamath":    _load_metamath,     # meta-math/MetaMathQA — math CoT, 395k (LLaMA-Pro SFT)
    "P2_evol_code":   _load_evol_code,    # theblackcat102/evol-codealpaca-v1 — Python instruct, 111k (LLaMA-Pro SFT)
    "P3_science":     _load_science,      # allenai/sciq — science QA, 13.7k
    "P4_medical":     _load_medical,      # openlifescienceai/medmcqa — medical MCQ, 182k
    # ── Retained alternatives (not in DEFAULT_PERIODS) ────────────────────────
    "P5_commonsense": _load_commonsense,  # tau/commonsense_qa — commonsense MCQ, 10.9k (not active)
    "P1_trivia":      _load_trivia,       # trivia_qa rc.nocontext — factual QA, 138k
    "P2_code":        _load_code,         # flytech/python-codes-25k — Python instruct, 25k
    # ── Retained P1 math alternatives (ablations only) ───────────────────────
    "P1_math":        _load_math,         # competition math — near-zero F1, kept for ablation only
    "P1_gsm8k":       _load_gsm8k,        # gsm8k — full CoT, 7.4k (size-constrained)
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_domain_sequential_periods(
    periods: Optional[List[str]] = None,
    n_per_period: int = 8_000,
    split_frac: float = 0.50,
    max_target_words: int = 200,
    seed: int = 42,
    num_proc: int = 4,
    **kwargs,
) -> Dict[str, Dataset]:
    """Load the Paper B five-domain curriculum and return period-keyed Datasets.

    All domains use natural Q/A framing (no arbitrary completion split).
    The code domain uses instruction→program framing; all others use
    question→answer framing.

    Parameters
    ----------
    periods          : list of domain period IDs (default: all four in canonical
                       order P1_metamath → P2_evol_code → P3_science → P4_medical)
    n_per_period     : maximum examples per domain after subsampling
    split_frac       : legacy parameter (unused — all domains have natural Q/A framing)
    max_target_words : legacy parameter (unused for natural-framing domains)
    seed             : reproducibility seed (affects shuffle + subsample)
    num_proc         : parallel workers for Dataset.map

    Returns
    -------
    Dict[period_id, Dataset]   columns: input_text (str), target_text (str), period (str)

    Examples
    --------
    >>> from data.domain_sequential import load_domain_sequential_periods
    >>> # All 4 domains:
    >>> periods = load_domain_sequential_periods(n_per_period=8_000)
    >>> list(periods.keys())
    ['P1_metamath', 'P2_evol_code', 'P3_science', 'P4_medical']
    >>> # Subset (e.g. for quick smoke tests):
    >>> periods = load_domain_sequential_periods(
    ...     periods=["P1_metamath", "P2_evol_code"], n_per_period=500
    ... )
    """
    if periods is None:
        periods = DEFAULT_PERIODS

    period_set = set(periods)
    unknown = period_set - set(_DOMAIN_LOADERS)
    if unknown:
        raise ValueError(
            f"Unknown period IDs: {sorted(unknown)}. "
            f"Valid IDs: {list(_DOMAIN_LOADERS)}"
        )

    result: Dict[str, Dataset] = {}

    # Iterate in canonical DEFAULT_PERIODS order for standard periods, then append
    # any non-default periods (e.g. P1_trivia fallback) in the order requested.
    _default_set = set(DEFAULT_PERIODS)
    _ordered = [p for p in DEFAULT_PERIODS if p in period_set] + \
               [p for p in periods        if p not in _default_set]

    for period_id in _ordered:

        loader = _DOMAIN_LOADERS[period_id]
        raw_ds = loader(n_per_period, seed)

        # All current loaders produce input_text / target_text directly.
        # The finalise() completion-split path is kept as a fallback for any
        # future loader that still returns a raw "text" column.
        if "input_text" in raw_ds.column_names:
            ds = raw_ds
            ds = ds.map(lambda _: {"period": period_id}, num_proc=num_proc,
                        desc=f"  {period_id}: add period label")
            ds = drop_short(ds, col="input_text",  min_len=20)
            ds = drop_short(ds, col="target_text", min_len=3)
            ds = subsample(ds, n_per_period, seed)
            final_ds = keep_columns(ds, STANDARD_COLS)
        else:
            final_ds = finalise(
                raw_ds,
                period=period_id,
                seed=seed,
                n=n_per_period,
                split_frac=split_frac,
                max_target_words=max_target_words,
                text_col="text",
                num_proc=num_proc,
            )

        result[period_id] = final_ds
        sample = final_ds[0]
        print(
            f"  {period_id}: {len(final_ds):,} examples  "
            f"(input ≈ {len(sample['input_text'].split()):,} words, "
            f"target ≈ {len(sample['target_text'].split()):,} words)"
        )

    return result
