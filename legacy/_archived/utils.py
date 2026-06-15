"""Self-contained period dataset loader for Phase 0.

Minimal JSONL reader that expects the following on-disk layout, produced by
the ``Phase0/data/download_*.py`` scripts::

    Phase0/data/processed/<dataset>/
      metadata.json                  # optional; may contain ordered timeline
      timeline.json                  # optional; ordered list of period ids
      stream/<period_id>.jsonl       # one line per doc: {"text": ...}
      probes/<period_id>.jsonl       # one line per probe; schema below

Probe schema (JSON-per-line)::

    {
      "question": "...",
      "evidence": "... (optional)",
      "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer_key": "A"
    }

Rules:

* If ``timeline.json`` or ``metadata.json.timeline`` is present, period ids
  are consumed in that order.
* Otherwise the loader sorts the ``stream/`` directory lexicographically,
  which is fine for zero-padded ids like ``2021-01``.
* Periods with empty streams are skipped.

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PeriodData:
    """One temporal period's raw documents + probes.

    The harness splits ``docs`` into train / eval using a seeded random
    shuffle so the same split is reproducible across baselines.
    """
    label: str
    index: int
    docs: List[Dict[str, Any]]
    probes: List[Dict[str, Any]]


# ── Readers ──────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _timeline_for(root: Path) -> List[str]:
    """Return the ordered list of period ids under ``root``.

    Preference: ``timeline.json`` > ``metadata.json["timeline"]`` >
    lexicographic listing of ``stream/*.jsonl``.
    """
    tl = root / "timeline.json"
    if tl.exists():
        return list(json.loads(tl.read_text()))
    meta = root / "metadata.json"
    if meta.exists():
        md = json.loads(meta.read_text())
        if "timeline" in md:
            return list(md["timeline"])
    stream_dir = root / "stream"
    if not stream_dir.exists():
        return []
    return sorted(p.stem for p in stream_dir.glob("*.jsonl"))


def load_dataset(
    data_root: str,
    max_periods: Optional[int] = None,
    max_docs_per_period: Optional[int] = None,
) -> List[PeriodData]:
    """Materialise every period under ``data_root`` in temporal order.

    ``max_periods`` and ``max_docs_per_period`` are soft caps for quick runs.
    Raises ``FileNotFoundError`` if ``data_root`` doesn't exist so baselines
    fail loudly rather than silently training on nothing.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Run the matching "
            f"Phase0/data/download_*.py script first."
        )

    timeline = _timeline_for(root)
    out: List[PeriodData] = []
    for pidx, pid in enumerate(timeline):
        if max_periods is not None and pidx >= max_periods:
            break
        docs = _read_jsonl(root / "stream" / f"{pid}.jsonl")
        if not docs:
            continue
        if max_docs_per_period:
            docs = docs[:max_docs_per_period]
        probes = _read_jsonl(root / "probes" / f"{pid}.jsonl")
        out.append(PeriodData(label=pid, index=len(out), docs=docs, probes=probes))
    return out


def available_datasets(data_dir: str) -> List[str]:
    """Return the list of dataset names prepared under ``data_dir``.

    Used by reports and scripts to avoid hard-coding dataset names.
    """
    d = Path(data_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "stream").exists())
