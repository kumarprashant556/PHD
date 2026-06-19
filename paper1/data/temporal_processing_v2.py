"""Version-2 temporal dataset processing for CAPSEL/INCA.

The processor reads raw temporal documents and writes two artifacts per period:

    processed/stream_v2/<period>.jsonl   — formatted training task examples
    processed/probes_v2/<period>.jsonl   — frozen evaluation probes

============================================================
STREAM — training task examples
============================================================
Every line is a ready-to-use text-to-text training example following the
temporal data system design in docs/dataset_strategy_FINAL.md:

  task="completion"
      Primary training signal.  Dense, smooth token-level loss that the
      saturation detector (RIR / grad-norm / CKA / plateau) can read cleanly.

      input : "period: 2018_H1\\ncomplete: <first 55 % of document words>"
      target: "<next 128 words>"

  task="salient_span_denoising"
      Auxiliary training signal.  Masks up to 2 salient spans (capitalised
      tokens or 4-digit years) per sentence using T5 <extra_id_N> sentinels.
      Fact-targeted masking (+6 LAMA / +5.8 temporal tasks; arXiv 2204.07994).

      input : "period: 2018_H1\\ndenoise: <corrupted sentence>"
      target: "<extra_id_0> span0 <extra_id_1> span1 <extra_id_2>"

Both tasks go in the same JSONL file.  Filter by task= at training time for
the E-FORMAT ablation (F-COMP = completion only, F-DENOISE = ssd only,
F-MIX = 70 % completion + 30 % salient-span denoising).

============================================================
PROBES — frozen evaluation items
============================================================
Three types, created once and re-run across future periods:

  completion   — prefix-to-continuation loss; sanity/style check only.
  entity_cloze — mask a named entity with <mask>; primary factual signal.
  date_cloze   — mask a 4-digit year with <mask>; temporal factual signal.

NOTE: salient_span_denoising is a TRAINING objective, NOT an eval probe.
      It has been moved out of probes_v2 and into stream_v2.

Key probe fields (see docs/dataset_strategy_FINAL.md §3):
  stability          "stable" | "updated" | "deprecated"
                     Defaults to "stable".  Update via a cross-period diff step
                     before Paper A experiments (D.1 task in TASKS.md).
  aliases            Alternate correct answers for EM scoring (empty list for
                     now; populate when NER quality improves — gap #8).
  eval_after_periods All periods from origin_period onward.  The evaluator
                     re-runs each probe after each future period to fill
                     the BWT regret matrix R[i,j] = model-after-i on probes-of-j.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CC_NEWS_RAW = REPO_ROOT / "local_data" / "cc_news" / "raw" / "raw.jsonl"
DEFAULT_TIC_LM_RAW_DIR = REPO_ROOT / "local_data" / "tic_lm" / "raw"

STREAM_DIRNAME = "stream_v2"
PROBES_DIRNAME = "probes_v2"

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}|\d{4}(?:-\d{2})?(?:-\d{2})?")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
)
_WS_RE = re.compile(r"\s+")

# Stop-words filtered out of entity cloze (sentence-initial / generic tokens).
_ENTITY_STOPWORDS = {
    "the", "this", "that", "these", "those", "he", "she", "it", "they",
    "we", "you", "his", "her", "its", "their", "our", "your",
    "mr", "mrs", "ms", "dr", "st", "inc", "llc", "ltd",
}


def clean_text(text: str) -> str:
    """Collapse whitespace without depending on HuggingFace datasets."""
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProcessingConfig:
    """Shared knobs for CC-News and TiC-LM preprocessing."""

    dataset: str
    raw_path: Path
    output_root: Path
    period_granularity: str = "half_year"
    max_docs_per_period: int = 0
    probes_per_period: int = 300
    min_words: int = 80
    min_chars: int = 300
    min_sentences: int = 3
    max_periods: int = 0
    seed: int = 42
    force: bool = False
    write_legacy_copy: bool = False
    stream_dirname: str = STREAM_DIRNAME
    probes_dirname: str = PROBES_DIRNAME


@dataclass
class PeriodBucket:
    # Intermediate raw docs (stored in memory; not written to disk directly).
    docs: List[Dict[str, Any]] = field(default_factory=list)
    seen_keys: set = field(default_factory=set)
    seen_count: int = 0
    dropped: int = 0


# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------

def process_cc_news_raw(
    raw_path: Path = DEFAULT_CC_NEWS_RAW,
    output_root: Path = REPO_ROOT / "local_data" / "cc_news" / "processed",
    period_granularity: str = "half_year",
    max_docs_per_period: int = 0,
    probes_per_period: int = 300,
    min_words: int = 80,
    min_chars: int = 300,
    min_sentences: int = 3,
    max_periods: int = 0,
    seed: int = 42,
    force: bool = False,
    write_legacy_copy: bool = False,
) -> Dict[str, Any]:
    """Process local CC-News raw JSONL into stream/probe period files."""
    cfg = ProcessingConfig(
        dataset="cc_news",
        raw_path=Path(raw_path),
        output_root=Path(output_root),
        period_granularity=period_granularity,
        max_docs_per_period=max_docs_per_period,
        probes_per_period=probes_per_period,
        min_words=min_words,
        min_chars=min_chars,
        min_sentences=min_sentences,
        max_periods=max_periods,
        seed=seed,
        force=force,
        write_legacy_copy=write_legacy_copy,
    )
    return _process_rows(cfg, _iter_cc_news_rows(cfg.raw_path))


def process_tic_lm_raw(
    raw_dir: Path = DEFAULT_TIC_LM_RAW_DIR,
    output_root: Path = REPO_ROOT / "local_data" / "tic_lm" / "processed",
    period_granularity: str = "day",
    max_docs_per_period: int = 0,
    probes_per_period: int = 300,
    min_words: int = 80,
    min_chars: int = 300,
    min_sentences: int = 3,
    max_periods: int = 0,
    seed: int = 42,
    force: bool = False,
    write_legacy_copy: bool = False,
) -> Dict[str, Any]:
    """Process local TiC-LM raw daily JSONL files into stream/probe periods."""
    cfg = ProcessingConfig(
        dataset="tic_lm",
        raw_path=Path(raw_dir),
        output_root=Path(output_root),
        period_granularity=period_granularity,
        max_docs_per_period=max_docs_per_period,
        probes_per_period=probes_per_period,
        min_words=min_words,
        min_chars=min_chars,
        min_sentences=min_sentences,
        max_periods=max_periods,
        seed=seed,
        force=force,
        write_legacy_copy=write_legacy_copy,
    )
    return _process_rows(cfg, _iter_tic_lm_rows(cfg.raw_path))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _process_rows(cfg: ProcessingConfig, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    _validate_config(cfg)
    stream_dir = cfg.output_root / cfg.stream_dirname
    probes_dir = cfg.output_root / cfg.probes_dirname
    meta_path = cfg.output_root / "metadata_v2.json"
    timeline_path = cfg.output_root / "timeline_v2.json"

    if not cfg.force and (timeline_path.exists() or stream_dir.exists() or probes_dir.exists()):
        raise FileExistsError(
            f"V2 processed output already exists under {cfg.output_root}. "
            "Pass --force to overwrite it."
        )

    _reset_dir(stream_dir)
    _reset_dir(probes_dir)

    rng = random.Random(cfg.seed)
    buckets: Dict[str, PeriodBucket] = {}
    total_seen = 0
    total_dropped = 0

    for raw in rows:
        total_seen += 1
        date = _normalise_date(str(raw.get("date") or raw.get("period") or ""))
        period = _period_from_date(date, cfg.period_granularity)
        if not period:
            total_dropped += 1
            continue

        text = clean_text(str(raw.get("text") or ""))
        title = clean_text(str(raw.get("title") or ""))
        url = str(raw.get("url") or "")
        if not _accept_document(text, cfg):
            total_dropped += 1
            continue

        bucket = buckets.setdefault(period, PeriodBucket())
        dedup_key = _dedup_key(url=url, title=title, text=text)
        if dedup_key in bucket.seen_keys:
            bucket.dropped += 1
            total_dropped += 1
            continue
        bucket.seen_keys.add(dedup_key)
        bucket.seen_count += 1

        # _make_raw_doc stores the document in memory for later training/probe generation.
        # The raw doc (with 'text') is NOT written to disk; only formatted examples are.
        doc = _make_raw_doc(
            dataset=cfg.dataset,
            period=period,
            date=date,
            text=text,
            title=title,
            url=url,
            raw_id=raw.get("doc_id") or raw.get("id"),
        )
        _reservoir_add(bucket.docs, doc, cfg.max_docs_per_period, bucket.seen_count, rng)

    timeline = sorted(buckets)
    if cfg.max_periods:
        timeline = timeline[: cfg.max_periods]

    counts: Dict[str, Dict[str, int]] = {}
    for period_idx, period in enumerate(timeline):
        # future_periods: from this period to end of timeline.
        # Used to populate eval_after_periods on every probe so the evaluator
        # knows which model checkpoints to re-run each probe against (BWT matrix).
        future_periods = timeline[period_idx:]

        docs = sorted(buckets[period].docs, key=lambda item: item["doc_id"])
        _rewrite_doc_ids(docs, cfg.dataset, period)

        # --- STREAM: formatted training examples (completion + salient-span) ---
        training_examples = _build_training_examples(docs, rng)

        # --- PROBES: frozen eval items (completion sanity + entity/date cloze) ---
        probes = _build_probes(docs, cfg.probes_per_period, rng, future_periods)

        n_stream = _write_jsonl(stream_dir / f"{period}.jsonl", training_examples)
        n_probes = _write_jsonl(probes_dir / f"{period}.jsonl", probes)
        counts[period] = {
            "stream_examples": n_stream,       # total training lines (comp + ssd)
            "stream_source_docs": len(docs),   # unique source documents sampled
            "probes": n_probes,
            "seen_before_sampling": buckets[period].seen_count,
            "dropped_after_period_bucket": buckets[period].dropped,
        }

    metadata = {
        "schema_version": "temporal_processing_v2",
        "dataset": cfg.dataset,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "raw_path": str(cfg.raw_path),
        "output_root": str(cfg.output_root),
        "stream_dir": str(stream_dir),
        "probes_dir": str(probes_dir),
        "timeline": timeline,
        "n_periods": len(timeline),
        "period_granularity": cfg.period_granularity,
        "max_docs_per_period": cfg.max_docs_per_period,
        "probes_per_period": cfg.probes_per_period,
        "filters": {
            "min_words": cfg.min_words,
            "min_chars": cfg.min_chars,
            "min_sentences": cfg.min_sentences,
        },
        "total_raw_rows_seen": total_seen,
        "total_rows_dropped": total_dropped,
        "counts": counts,
        "formats": {
            "stream": {
                "tasks": ["completion", "salient_span_denoising"],
                "fields": ["task", "doc_id", "period", "source", "date", "input", "target"],
                "note": (
                    "Each line is a ready-to-use training example. "
                    "Filter by task= for E-FORMAT ablation: "
                    "F-COMP (completion only), F-DENOISE (ssd only), "
                    "F-MIX (70% completion + 30% salient_span_denoising)."
                ),
            },
            "probes": {
                "types": ["completion", "entity_cloze", "date_cloze"],
                "fields": [
                    "probe_id", "probe_type", "origin_period", "source",
                    "source_doc_id", "date", "input", "target", "answer",
                    "aliases", "answer_type", "stability", "evidence",
                    "eval_after_periods",
                ],
                "stability_values": ["stable", "updated", "deprecated"],
                "note": (
                    "Probes are frozen at creation. Re-evaluate after each future "
                    "period to fill the BWT regret matrix R[i,j]. "
                    "stability defaults to 'stable'; update via cross-period diff "
                    "before Paper A experiments (D.1 / D.2 in TASKS.md). "
                    "salient_span_denoising is a training format — NOT a probe type."
                ),
            },
        },
    }

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if cfg.write_legacy_copy:
        _write_legacy_copy(cfg.output_root, stream_dir, probes_dir, timeline, metadata)

    return metadata


# ---------------------------------------------------------------------------
# Raw-row iterators (dataset-specific)
# ---------------------------------------------------------------------------

def _iter_cc_news_rows(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CC-News raw file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_tic_lm_rows(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"TiC-LM raw directory not found: {raw_dir}")
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No TiC-LM raw JSONL files found in: {raw_dir}")
    for path in files:
        fallback_date = path.stem
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row.setdefault("date", fallback_date)
                yield row


# ---------------------------------------------------------------------------
# Config validation and directory helpers
# ---------------------------------------------------------------------------

def _validate_config(cfg: ProcessingConfig) -> None:
    valid = {"day", "month", "quarter", "half_year", "year"}
    if cfg.period_granularity not in valid:
        raise ValueError(f"period_granularity must be one of {sorted(valid)}")
    if cfg.max_docs_per_period < 0:
        raise ValueError("max_docs_per_period must be >= 0")
    if cfg.probes_per_period < 0:
        raise ValueError("probes_per_period must be >= 0")


def _reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.jsonl"):
        child.unlink()


# ---------------------------------------------------------------------------
# Document filtering, normalisation, dedup
# ---------------------------------------------------------------------------

def _accept_document(text: str, cfg: ProcessingConfig) -> bool:
    if len(text) < cfg.min_chars:
        return False
    words = text.split()
    if len(words) < cfg.min_words:
        return False
    if len(_sentences(text)) < cfg.min_sentences:
        return False
    return True


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _normalise_date(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("period_"):
        raw = raw[len("period_"):]
    match = re.search(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", raw)
    if not match:
        return ""
    year = match.group(1)
    month = match.group(2) or "01"
    day = match.group(3)
    return f"{year}-{month}-{day}" if day else f"{year}-{month}"


def _period_from_date(date: str, granularity: str) -> str:
    match = re.match(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$", date)
    if not match:
        return ""
    year = int(match.group(1))
    month = int(match.group(2))
    day = match.group(3)
    if not 1 <= month <= 12:
        return ""
    if granularity == "day":
        return f"period_{year:04d}-{month:02d}-{int(day or 1):02d}"
    if granularity == "month":
        return f"{year:04d}-{month:02d}"
    if granularity == "quarter":
        return f"{year:04d}_Q{((month - 1) // 3) + 1}"
    if granularity == "half_year":
        return f"{year:04d}_H{1 if month <= 6 else 2}"
    if granularity == "year":
        return f"{year:04d}"
    return ""


def _dedup_key(url: str, title: str, text: str) -> str:
    if url:
        return "url:" + url.strip().lower()
    if title:
        return "title:" + title.strip().lower()[:160]
    digest = hashlib.sha1(text[:1000].encode("utf-8")).hexdigest()
    return "text:" + digest


# ---------------------------------------------------------------------------
# Raw intermediate document
# Stored in PeriodBucket.docs in memory; NOT written to disk.
# Passed into _build_training_examples and _build_probes.
# ---------------------------------------------------------------------------

def _make_raw_doc(
    dataset: str,
    period: str,
    date: str,
    text: str,
    title: str,
    url: str,
    raw_id: Any,
) -> Dict[str, Any]:
    digest = hashlib.sha1((url or title or text[:300]).encode("utf-8")).hexdigest()[:12]
    doc: Dict[str, Any] = {
        "text": text,
        "doc_id": f"{dataset}_{period}_{digest}",
        "period": period,
        "source": dataset,
        "date": date,
        "char_len": len(text),
        "word_count": len(text.split()),
    }
    if title:
        doc["title"] = title
    if url:
        doc["url"] = url
    if raw_id:
        doc["raw_id"] = raw_id
    return doc


def _reservoir_add(
    docs: List[Dict[str, Any]],
    doc: Dict[str, Any],
    max_docs: int,
    seen_count: int,
    rng: random.Random,
) -> None:
    if max_docs == 0:
        docs.append(doc)
        return
    if len(docs) < max_docs:
        docs.append(doc)
        return
    idx = rng.randint(1, seen_count)
    if idx <= max_docs:
        docs[idx - 1] = doc


def _rewrite_doc_ids(docs: List[Dict[str, Any]], dataset: str, period: str) -> None:
    for idx, doc in enumerate(docs):
        doc["doc_id"] = f"{dataset}_{period}_{idx:06d}"


# ---------------------------------------------------------------------------
# STREAM: formatted training examples
# ---------------------------------------------------------------------------

def _build_training_examples(
    docs: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Return formatted training examples for the stream file.

    Each document produces:
      - 1 completion example  (task="completion")          — always, if doc >= 120 words
      - 1 salient-span denoising example (task="salient_span_denoising") — if a valid
        sentence with 2+ salient spans exists

    Both tasks are interleaved in the same JSONL so the trainer can filter by
    task= for the E-FORMAT ablation without re-processing.
    """
    examples: List[Dict[str, Any]] = []
    for doc in docs:
        comp = _make_completion_training_ex(doc)
        if comp:
            examples.append(comp)
        ssd = _make_ssd_training_ex(doc, rng)
        if ssd:
            examples.append(ssd)
    return examples


def _make_completion_training_ex(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Primary training signal: period-prefixed prefix-to-continuation completion.

    input : "period: <period>\\ncomplete: <first 55% of words>"
    target: "<next 128 words>"
    """
    words = doc["text"].split()
    if len(words) < 120:
        return None
    split = max(40, int(len(words) * 0.55))
    target_words = words[split: split + 128]
    if len(target_words) < 24:
        return None
    ex: Dict[str, Any] = {
        "doc_id": doc["doc_id"] + "_comp",
        "period": doc["period"],
        "source": doc["source"],
        "date": doc.get("date", ""),
        "task": "completion",
        "input": "period: " + doc["period"] + "\ncomplete: " + " ".join(words[:split]),
        "target": " ".join(target_words),
    }
    if "title" in doc:
        ex["title"] = doc["title"]
    if "url" in doc:
        ex["url"] = doc["url"]
    return ex


def _make_ssd_training_ex(
    doc: Dict[str, Any], rng: random.Random
) -> Optional[Dict[str, Any]]:
    """Auxiliary training signal: salient-span denoising (T5 <extra_id_N> style).

    Masks 2 salient spans (capitalised multi-char tokens or 4-digit years) per
    sentence.  This example goes into the STREAM (training), NOT the probes file.

    input : "period: <period>\\ndenoise: <corrupted sentence>"
    target: "<extra_id_0> span0 <extra_id_1> span1 <extra_id_2>"
    """
    for sentence in _sentences(doc["text"]):
        spans = _TOKEN_RE.findall(sentence)
        salient = [
            span
            for span in spans
            if re.match(r"^(?:[A-Z][A-Za-z\-']{2,}|\d{4})", span)
        ]
        salient = _unique_preserve_order(salient)
        if len(salient) < 2:
            continue
        chosen = salient[:2]
        corrupted = sentence
        target_parts: List[str] = []
        for idx, span in enumerate(chosen):
            token = f"<extra_id_{idx}>"
            corrupted = re.sub(rf"\b{re.escape(span)}\b", token, corrupted, count=1)
            target_parts.extend([token, span])
        target_parts.append(f"<extra_id_{len(chosen)}>")
        ex: Dict[str, Any] = {
            "doc_id": doc["doc_id"] + "_ssd",
            "period": doc["period"],
            "source": doc["source"],
            "date": doc.get("date", ""),
            "task": "salient_span_denoising",
            "input": "period: " + doc["period"] + "\ndenoise: " + corrupted,
            "target": " ".join(target_parts),
            "evidence": sentence,
        }
        if "title" in doc:
            ex["title"] = doc["title"]
        if "url" in doc:
            ex["url"] = doc["url"]
        return ex
    return None


# ---------------------------------------------------------------------------
# PROBES: frozen evaluation items
# ---------------------------------------------------------------------------

def _build_probes(
    docs: List[Dict[str, Any]],
    probes_per_period: int,
    rng: random.Random,
    future_periods: List[str],
) -> List[Dict[str, Any]]:
    """Build eval probes: completion (sanity), entity_cloze, date_cloze.

    salient_span_denoising has been moved to the training stream.  Probe types
    here are evaluation-only items that get re-run across future periods for the
    BWT regret matrix.

    Args:
        future_periods: timeline[period_idx:] — periods from origin period onwards.
                        Stored as eval_after_periods on each probe.
    """
    if probes_per_period == 0:
        return []
    probes: List[Dict[str, Any]] = []
    per_doc_builders = [
        _make_completion_probe,
        _make_entity_cloze_probe,
        _make_date_cloze_probe,
    ]
    candidates = docs[:]
    rng.shuffle(candidates)
    for doc in candidates:
        for builder in per_doc_builders:
            probe = builder(doc, rng, future_periods)
            if probe:
                probes.append(probe)
                if len(probes) >= probes_per_period:
                    return probes
    return probes


def _base_probe(
    doc: Dict[str, Any],
    probe_type: str,
    future_periods: List[str],
) -> Dict[str, Any]:
    """Shared fields for all probe types.

    stability         defaults to "stable".  A cross-period diff step (D.1 / D.2
                      in TASKS.md) should update entity/date cloze probes to
                      "updated" or "deprecated" once later periods are available.
    aliases           empty list for now; populate when NER quality improves.
    eval_after_periods all periods from origin_period onwards so the evaluator
                      can reconstruct the full R[i,j] regret matrix.
    """
    return {
        "probe_id": f"{doc['doc_id']}_{probe_type}",
        "probe_type": probe_type,
        "origin_period": doc["period"],
        "source": doc["source"],
        "source_doc_id": doc["doc_id"],
        "date": doc.get("date", ""),
        "stability": "stable",
        "aliases": [],
        "eval_after_periods": future_periods,
    }


def _make_completion_probe(
    doc: Dict[str, Any],
    rng: random.Random,
    future_periods: List[str],
) -> Optional[Dict[str, Any]]:
    """Completion probe: sanity/style check (not the headline BWT metric)."""
    words = doc["text"].split()
    if len(words) < 120:
        return None
    split = max(40, int(len(words) * 0.55))
    target = words[split: split + 96]
    if len(target) < 24:
        return None
    probe = _base_probe(doc, "completion", future_periods)
    probe.update(
        {
            "input": "period: " + doc["period"] + "\ncomplete: " + " ".join(words[:split]),
            "target": " ".join(target),
            "answer": " ".join(target),
            "answer_type": "continuation",
            "evidence": doc["text"][:1200],
        }
    )
    return probe


def _make_entity_cloze_probe(
    doc: Dict[str, Any],
    rng: random.Random,
    future_periods: List[str],
) -> Optional[Dict[str, Any]]:
    """Entity cloze probe: mask a named entity with <mask> (primary factual signal)."""
    for sentence in _sentences(doc["text"]):
        entities = [
            ent.strip()
            for ent in _ENTITY_RE.findall(sentence)
            if (
                len(ent.strip()) >= 3
                and ent.strip().lower() not in _ENTITY_STOPWORDS
            )
        ]
        entities = _unique_preserve_order(entities)
        if not entities:
            continue
        answer = entities[-1]
        masked = re.sub(rf"\b{re.escape(answer)}\b", "<mask>", sentence, count=1)
        if "<mask>" not in masked:
            continue
        probe = _base_probe(doc, "entity_cloze", future_periods)
        probe.update(
            {
                "input": f"period: {doc['period']}\nfill: {masked}",
                "target": answer,
                "answer": answer,
                "aliases": [],
                "answer_type": "entity",
                "evidence": sentence,
            }
        )
        return probe
    return None


def _make_date_cloze_probe(
    doc: Dict[str, Any],
    rng: random.Random,
    future_periods: List[str],
) -> Optional[Dict[str, Any]]:
    """Date cloze probe: mask a 4-digit year with <mask> (temporal factual signal)."""
    for sentence in _sentences(doc["text"]):
        dates = re.findall(r"\b(?:19|20)\d{2}\b", sentence)
        if not dates:
            continue
        answer = dates[-1]
        masked = re.sub(rf"\b{re.escape(answer)}\b", "<mask>", sentence, count=1)
        probe = _base_probe(doc, "date_cloze", future_periods)
        probe.update(
            {
                "input": f"period: {doc['period']}\nfill: {masked}",
                "target": answer,
                "answer": answer,
                "aliases": [],
                "answer_type": "date",
                "evidence": sentence,
            }
        )
        return probe
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: set = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_legacy_copy(
    output_root: Path,
    stream_dir: Path,
    probes_dir: Path,
    timeline: List[str],
    metadata: Dict[str, Any],
) -> None:
    legacy_stream = output_root / "stream"
    legacy_probes = output_root / "probes"
    _reset_dir(legacy_stream)
    _reset_dir(legacy_probes)
    for src in stream_dir.glob("*.jsonl"):
        (legacy_stream / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for src in probes_dir.glob("*.jsonl"):
        (legacy_probes / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (output_root / "timeline.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    legacy_meta = dict(metadata)
    legacy_meta["copied_from"] = "temporal_processing_v2"
    (output_root / "metadata.json").write_text(
        json.dumps(legacy_meta, indent=2), encoding="utf-8"
    )
