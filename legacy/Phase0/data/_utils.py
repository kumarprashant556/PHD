"""Shared helpers for Phase 0 dataset downloaders.

Every downloader in this folder writes a two-tier layout:

    Phase0/data/raw/<dataset>/        # original dataset files (Parquet / JSONL)
    Phase0/data/processed/<dataset>/  # period-sliced JSONL the loader reads
        timeline.json                   # ordered list of period ids
        metadata.json                   # source, period scheme, counts
        stream/<period_id>.jsonl        # one line per doc  — {"text":…, "doc_id":…, …}
        probes/<period_id>.jsonl        # one line per probe — mixed formats (see below)

The raw/ directory holds the original dataset as it came from HuggingFace —
Parquet files for structured datasets, JSONL for streamed ones.  HuggingFace's
own Arrow cache is intentionally left in its default location (~/.cache/
huggingface) and is NOT stored here; raw/ is for the actual data.

Probe formats
-------------
Each probe carries a "format" field so a single loader can handle all
datasets without branching on the dataset name:

  "mc4"        — 4-way multiple-choice (question / choices / answer_key)
  "completion" — next-span prediction   (prompt / completion)
  "instruction"— instruction-following  (instruction / response)  [TRACE]

All helpers here are deliberately framework-light so a downloader is a
short, readable script.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent          # Phase0/
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
RAW_ROOT = REPO_ROOT / "data" / "raw"

# Treat "no cap" sentinels uniformly across downloaders.
# A value of 0 from the CLI means "all"; we convert that once via cap().
NO_CAP = 10 ** 12


def dataset_dir(name: str) -> Path:
    """Processed JSONL output dir (created if missing)."""
    out = PROCESSED_ROOT / name
    (out / "stream").mkdir(parents=True, exist_ok=True)
    (out / "probes").mkdir(parents=True, exist_ok=True)
    return out


def raw_dir(name: str) -> Path:
    """Per-dataset raw data directory (created if missing).

    This is where the original dataset files are saved — Parquet splits for
    structured HuggingFace datasets, JSONL for streamed ones.

    HuggingFace's internal Arrow cache is NOT stored here; it stays in its
    default location (~/.cache/huggingface).  Do NOT pass the path returned
    by this function as ``cache_dir`` to ``load_dataset``.
    """
    out = RAW_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_raw_parquet(dataset, out: Path) -> None:
    """Export every split of a HuggingFace DatasetDict to Parquet files.

    Writes one file per split: ``<out>/<split>.parquet``.
    Safe to call with a plain Dataset (non-dict) — written as ``train.parquet``.
    Silently skips IterableDataset (streaming) — save_raw_jsonl handles those.
    """
    try:
        import datasets as _hf  # type: ignore
        if isinstance(dataset, _hf.DatasetDict):
            for split_name, split_ds in dataset.items():
                dest = out / f"{split_name}.parquet"
                split_ds.to_parquet(str(dest))
                mb = dest.stat().st_size / 1024 / 1024
                print(f"  [raw] {split_name}.parquet  ({mb:.1f} MB) → {dest}")
        elif isinstance(dataset, _hf.Dataset):
            dest = out / "train.parquet"
            dataset.to_parquet(str(dest))
            mb = dest.stat().st_size / 1024 / 1024
            print(f"  [raw] train.parquet  ({mb:.1f} MB) → {dest}")
        else:
            # IterableDataset or unknown — caller handles separately
            pass
    except Exception as e:
        print(f"  [raw] Parquet export error: {e}")


def save_raw_jsonl(rows: Iterable[Dict[str, Any]], out: Path, filename: str = "raw.jsonl") -> int:
    """Write a collection of raw rows as JSONL to raw/<dataset>/<filename>.

    Used for streaming datasets where Parquet export isn't feasible.
    Returns number of rows written.
    """
    dest = out / filename
    n = write_jsonl(dest, rows)
    mb = dest.stat().st_size / 1024 / 1024
    print(f"  [raw] {filename}  ({n:,} rows, {mb:.1f} MB) → {dest}")
    return n


def cap(value: Optional[int]) -> int:
    """Normalise a ``--max_*`` CLI flag: 0 / None means "no cap"."""
    if not value:
        return NO_CAP
    return int(value)


# ── Stream document builder ──────────────────────────────────────────────────

def make_doc(
    text: str,
    period: str = "",
    source: str = "",
    doc_idx: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standard stream document with training + provenance metadata."""
    parts = [p for p in [source, period] if p]
    doc_id = "_".join(parts + [f"{doc_idx:06d}"]) if parts else f"doc_{doc_idx:06d}"
    doc: Dict[str, Any] = {
        "text": text,
        "doc_id": doc_id,
        "period": period,
        "source": source,
        "char_len": len(text),
    }
    if extra:
        doc.update(extra)
    return doc


# ── Probe builders ────────────────────────────────────────────────────────────

def make_cloze_probe(
    text: str,
    max_choices: int = 4,
    period: str = "",
    source: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a 4-way cloze (MC) probe by masking a content word in ``text``."""
    _TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
    tokens = _TOKEN_RE.findall(text)
    seen, ordered = set(), []
    for t in tokens:
        low = t.lower()
        if low not in seen:
            seen.add(low)
            ordered.append(t)
    if len(ordered) < max_choices + 1:
        return None

    answer = None
    for t in ordered:
        if len(t) >= 5:
            answer = t
            break
    if answer is None:
        answer = ordered[0]

    distractors = [t for t in ordered if t.lower() != answer.lower()][: max_choices - 1]
    if len(distractors) < max_choices - 1:
        return None

    masked = re.sub(rf"\b{re.escape(answer)}\b", "____", text, count=1)
    if "____" not in masked:
        return None

    choices = [answer] + distractors
    keys = ["A", "B", "C", "D"]
    return {
        "format": "mc4",
        "question": masked,
        "evidence": "",
        "choices": {keys[i]: choices[i] for i in range(max_choices)},
        "answer_key": "A",
        "period": period,
        "source": source,
    }


def make_open_qa_probe(
    text: str,
    period: str = "",
    source: str = "",
    date: str = "",
) -> Optional[Dict[str, Any]]:
    """Build an open-answer QA probe by extracting a factual sentence.

    Heuristic: find a sentence with a proper noun (Title Case word or
    all-caps acronym) and turn it into an extractive QA pair.
    Input text = evidence article/passage.
    Returns None if no suitable sentence is found.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        if len(sent) < 40:
            continue
        # Find words that look like proper nouns / named entities
        nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", sent)
        if len(nouns) < 2:
            continue
        # Pick the last proper noun as the answer
        answer = nouns[-1]
        # Mask it with a question
        question = re.sub(rf"\b{re.escape(answer)}\b",
                          "what entity", sent, count=1, flags=re.IGNORECASE)
        question = f"According to the passage: {question}?"
        return {
            "question":   question.strip(),
            "answer":     answer.strip(),
            "choices":    {},         # open-answer → no choices
            "answer_key": "",
            "evidence":   sent.strip(),
            "date":       date,
            "period":     period,
            "source":     source,
        }
    return None


def make_completion_probe(
    text: str,
    split_ratio: float = 0.70,
    period: str = "",
    source: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a next-span completion probe by splitting a document."""
    if len(text) < 200:
        return None
    split_at = int(len(text) * split_ratio)
    while split_at < len(text) and text[split_at] not in " \n\t":
        split_at += 1
    prompt = text[:split_at].strip()
    completion = text[split_at:].strip()
    if len(prompt) < 80 or len(completion) < 30:
        return None
    return {
        "format": "completion",
        "prompt": prompt,
        "completion": completion,
        "period": period,
        "source": source,
    }


def make_instruction_probe(
    instruction: str,
    response: str,
    period: str = "",
    source: str = "",
) -> Optional[Dict[str, Any]]:
    """Build an instruction-following probe (Alpaca style)."""
    if not instruction.strip() or not response.strip():
        return None
    return {
        "format": "instruction",
        "instruction": instruction.strip(),
        "response": response.strip(),
        "period": period,
        "source": source,
    }


# ── I/O helpers ──────────────────────────────────────────────────────────────

def iter_raw_jsonl(path: Path):
    """Yield dicts from a JSONL file one line at a time (memory-efficient).

    Suitable for large files (e.g. raw/cc_news/raw.jsonl at 1.6 GB) because
    it never loads the whole file into memory.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write an iterable of dicts to a JSONL file.  Returns the row count."""
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_timeline(
    out: Path,
    period_ids: List[str],
    meta_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``timeline.json`` and ``metadata.json`` to *out*."""
    (out / "timeline.json").write_text(json.dumps(period_ids, indent=2))
    md: Dict[str, Any] = {"timeline": period_ids, "n_periods": len(period_ids)}
    if meta_extra:
        md.update(meta_extra)
    (out / "metadata.json").write_text(json.dumps(md, indent=2))


# ── HuggingFace import shim ──────────────────────────────────────────────────

def require_hf_datasets():
    """Lazy import so users without ``datasets`` get a clear error message."""
    try:
        import datasets as hf_datasets  # type: ignore
        return hf_datasets
    except ImportError as e:
        raise SystemExit(
            "This downloader needs the HuggingFace `datasets` library.\n"
            "Install it with:  pip install datasets"
        ) from e
