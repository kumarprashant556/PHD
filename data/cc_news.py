"""CC-News period loader  (data/cc_news.py)

Primary dataset for Phase 1 INCA training.

Source (primary)  : datasets/cc_news/raw/raw.jsonl              ← local, instant
Source (fallback) : vblagoje/cc_news on HuggingFace             ← downloaded on first use
Coverage          : ~630K English news articles, 2017–2019
Period key        : "YYYY_H{1|2}"  — half-year buckets
Default           : 4 usable periods: 2017_H1, 2017_H2, 2018_H1, 2018_H2
                    (2019_H1/H2 have only 1 article each — excluded by default)

Temporal split
--------------
2017_H1  Trump inauguration, early admin, travel ban
2017_H2  Tax reform, Harvey/Irma/Maria, Las Vegas
2018_H1  Trade war begins, Cambridge Analytica, #MeToo verdict
2018_H2  Midterms, Khashoggi, crypto crash

Usage
-----
from data.cc_news import load_cc_news_periods

periods = load_cc_news_periods(n_per_period=20_000)
# {"2017_H1": Dataset(input_text, target_text, period), ...}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Guard against the local datasets/ folder shadowing HuggingFace's `datasets` package.
# Without this, `from datasets import Dataset` silently imports the wrong module.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_sp_backup = sys.path[:]
sys.path = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)]
from datasets import Dataset  # noqa: E402
sys.path[:] = _sp_backup
del _sp_backup, _REPO_ROOT

from ._base import finalise, clean_text

# ── Default period layout ─────────────────────────────────────────────────────

# Only include periods with ≥1 000 articles (2019_H* have only 1 each)
DEFAULT_PERIODS: List[str] = [
    "2017_H1", "2017_H2",
    "2018_H1", "2018_H2",
]

# Path to the locally cached raw JSONL  (datasets/cc_news/raw.jsonl at repo root)
_LOCAL_JSONL = Path(__file__).resolve().parent.parent / "datasets" / "cc_news" / "raw" / "raw.jsonl"
_PROCESSED_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "cc_news" / "processed"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ym_to_period(date_str: str) -> str:
    """'2017-06' or '2017-06-15' → '2017_H1'."""
    try:
        parts = date_str.split("-")
        year, month = int(parts[0]), int(parts[1])
        return f"{year}_{'H1' if month <= 6 else 'H2'}"
    except Exception:
        return "unknown"


def _load_from_local(
    periods: List[str],
    n_per_period: int,
    seed: int,
) -> Dict[str, List[dict]]:
    """Read raw.jsonl and bucket into requested periods."""
    buckets: Dict[str, List[dict]] = {p: [] for p in periods}
    period_set = set(periods)

    with open(_LOCAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            period = _ym_to_period(obj.get("date", ""))
            if period not in period_set:
                continue
            text = clean_text(obj.get("text") or "")
            if text:
                buckets[period].append({"text": text, "period": period})

    # Shuffle + cap each period
    rng = random.Random(seed)
    for period in periods:
        rng.shuffle(buckets[period])
        buckets[period] = buckets[period][:n_per_period]

    return buckets


def _load_from_hf(
    periods: List[str],
    n_per_period: int,
    seed: int,
    cache_dir: Optional[str],
    num_proc: int,
) -> Dict[str, List[dict]]:
    """Fallback: download from HuggingFace and bucket."""
    from datasets import load_dataset

    print("  Downloading vblagoje/cc_news from HuggingFace …")
    ds = load_dataset("vblagoje/cc_news", split="train", cache_dir=cache_dir)

    period_set = set(periods)
    buckets: Dict[str, List[dict]] = {p: [] for p in periods}

    for row in ds:
        period = _ym_to_period(row.get("date", ""))
        if period not in period_set:
            continue
        text = clean_text(row.get("text") or "")
        if text:
            buckets[period].append({"text": text, "period": period})

    rng = random.Random(seed)
    for period in periods:
        rng.shuffle(buckets[period])
        buckets[period] = buckets[period][:n_per_period]

    return buckets


def _load_from_processed(
    stream_dir: Path,
    periods: List[str],
    n_per_period: int,
    seed: int,
) -> Dict[str, List[dict]]:
    """Read preprocessed stream JSONL files for requested periods (legacy v1: raw text)."""
    buckets: Dict[str, List[dict]] = {}
    rng = random.Random(seed)
    for period in periods:
        path = stream_dir / f"{period}.jsonl"
        rows: List[dict] = []
        if not path.exists():
            buckets[period] = rows
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = clean_text(obj.get("text") or "")
                if text:
                    rows.append({"text": text, "period": period})
        rng.shuffle(rows)
        buckets[period] = rows[:n_per_period]
    return buckets


def _load_v2_stream(
    stream_v2_dir: Path,
    periods: List[str],
    n_per_period: int,
    seed: int,
) -> Dict[str, List[dict]]:
    """Read v2 preprocessed stream JSONL files (already formatted as input/target pairs).

    v2 format (written by data/temporal_processing_v2.py):
        {"task": "completion"|"salient_span_denoising",
         "input": "period: 2017_H1\\ncomplete: ...",
         "target": "...",
         "period": "2017_H1", ...}

    Both task types are returned — completion (primary) and salient_span_denoising
    (auxiliary denoising objective).  No further text-splitting is applied.
    """
    buckets: Dict[str, List[dict]] = {}
    rng = random.Random(seed)
    for period in periods:
        path = stream_v2_dir / f"{period}.jsonl"
        rows: List[dict] = []
        if not path.exists():
            buckets[period] = rows
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inp = (obj.get("input") or "").strip()
                tgt = (obj.get("target") or "").strip()
                if inp and tgt:
                    rows.append({
                        "input_text": inp,
                        "target_text": tgt,
                        "period":      obj.get("period", period),
                        "task":        obj.get("task", "completion"),
                    })
        rng.shuffle(rows)
        buckets[period] = rows[:n_per_period]
    return buckets


def _processed_periods(stream_dir: Path) -> List[str]:
    if not stream_dir.exists():
        return []
    return sorted(path.stem for path in stream_dir.glob("*.jsonl") if path.stat().st_size > 0)


# ── Public API ────────────────────────────────────────────────────────────────

def load_cc_news_periods(
    periods: Optional[List[str]] = None,
    n_per_period: int = 20_000,
    split_frac: float = 0.50,
    max_target_words: int = 200,
    seed: int = 42,
    num_proc: int = 4,
    cache_dir: Optional[str] = None,
    processed_version: Optional[str] = None,
) -> Dict[str, Dataset]:
    """Load CC-News and return half-year temporal period datasets.

    Source priority (highest → lowest):
      1. datasets/cc_news/processed/stream_v2/  ← preferred: already-formatted
         task/input/target examples (completion + salient_span_denoising).
         Auto-detected when the directory exists; or forced with
         processed_version="v2".
      2. datasets/cc_news/raw/raw.jsonl          ← local raw cache
      3. vblagoje/cc_news on HuggingFace         ← downloaded on first use

    Parameters
    ----------
    periods          : half-year period keys to return (default: auto from stream_v2,
                       else 4 periods 2017–2018)
    n_per_period     : max training examples per period (randomly sampled if exceeded)
    split_frac       : (raw path only) encoder input fraction of words
    max_target_words : (raw path only) hard cap on decoder target length in words
    seed             : reproducibility seed
    num_proc         : parallel workers for Dataset.map (raw path only)
    cache_dir        : HuggingFace cache dir (HF fallback only)
    processed_version: "v2" forces the stream_v2 path; None = auto-detect

    Returns
    -------
    Dict[period_id, Dataset]  columns: input_text, target_text, period[, task]
    """
    stream_v2_dir = _PROCESSED_ROOT / "stream_v2"

    # ── Path 1: v2 preprocessed stream (auto-detected or forced) ─────
    use_v2 = (
        processed_version == "v2"
        or (
            processed_version is None
            and stream_v2_dir.exists()
            and any(stream_v2_dir.glob("*.jsonl"))
        )
    )

    if use_v2:
        available = _processed_periods(stream_v2_dir)
        if not available:
            raise FileNotFoundError(
                f"No processed v2 CC-News files found in {stream_v2_dir}.\n"
                "Run:  python scripts/process_temporal_data_v2.py cc_news --force"
            )
        if periods is None:
            # Prefer the well-populated default periods; fall back to all available
            periods = [p for p in DEFAULT_PERIODS if p in available] or available

        print(f"CC-News v2: reading preprocessed stream  ({stream_v2_dir})")
        buckets_v2 = _load_v2_stream(stream_v2_dir, periods, n_per_period, seed)

        result: Dict[str, Dataset] = {}
        for period in periods:
            rows = buckets_v2.get(period, [])
            if not rows:
                print(f"  WARNING: no v2 examples for {period} — skipping")
                continue
            ds = Dataset.from_list(rows)
            n_comp = sum(1 for r in rows if r.get("task") == "completion")
            n_ssd  = sum(1 for r in rows if r.get("task") == "salient_span_denoising")
            print(
                f"  {period}: {len(ds):,} training examples  "
                f"({n_comp:,} completion + {n_ssd:,} ssd)  [v2]"
            )
            result[period] = ds
        return result

    # ── Path 2 / 3: legacy raw text → apply completion framing ───────
    if periods is None:
        periods = DEFAULT_PERIODS

    if processed_version and processed_version != "v2":
        stream_name = "stream"
        stream_dir  = _PROCESSED_ROOT / stream_name
        available   = _processed_periods(stream_dir)
        if not available:
            raise FileNotFoundError(
                f"No processed CC-News files found in {stream_dir}."
            )
        if periods is None:
            periods = available
        print(f"CC-News: reading processed stream ({stream_dir})")
        buckets = _load_from_processed(stream_dir, periods, n_per_period, seed)
    elif _LOCAL_JSONL.exists():
        print(f"CC-News: reading local raw cache  ({_LOCAL_JSONL})")
        buckets = _load_from_local(periods, n_per_period, seed)
    else:
        print(f"CC-News: local cache not found at {_LOCAL_JSONL}")
        buckets = _load_from_hf(periods, n_per_period, seed, cache_dir, num_proc)

    # Apply completion framing (raw text → input_text / target_text)
    result = {}
    for period in periods:
        rows = buckets[period]
        if not rows:
            print(f"  WARNING: no articles found for {period} — skipping")
            continue
        raw_ds   = Dataset.from_list(rows)
        final_ds = finalise(
            raw_ds,
            period=period,
            seed=seed,
            n=n_per_period,
            split_frac=split_frac,
            max_target_words=max_target_words,
            text_col="text",
            num_proc=num_proc,
        )
        result[period] = final_ds
        print(
            f"  {period}: {len(final_ds):,} articles  "
            f"(input ≈ {len(final_ds[0]['input_text'].split())} words, "
            f"target ≈ {len(final_ds[0]['target_text'].split())} words)"
        )
    return result


# ── Convenience: statistics ───────────────────────────────────────────────────

def cc_news_stats(periods: Dict[str, Dataset]) -> None:
    """Print a quick per-period summary."""
    print(f"\nCC-News ({len(periods)} periods loaded):")
    for pid, ds in sorted(periods.items()):
        s = ds[0]
        print(f"  {pid:10s}  {len(ds):6,} docs  "
              f"~{len(s['input_text'].split()):4d}w input  "
              f"~{len(s['target_text'].split()):4d}w target")
