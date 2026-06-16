"""Probe-leakage analyzer for CC-News v2 — read-only, GPU-free.

Safe to run alongside an in-progress training sweep: only reads JSONL files
under datasets/cc_news/processed/, no torch import, no model load.

Question answered
-----------------
For each (after-period t, probes-period j) cell of the regret matrix, what
fraction of period-j probe answers has the model ALREADY seen during training
in periods 0..t?

If this leakage is high, the lack of forgetting in the regret matrix isn't
forgetting suppression — it's that there was nothing to forget, because the
"answer" keeps re-appearing in subsequent periods' training data. This is the
key sanity check for the flat-BWT finding on CC-News.

Two metrics are reported:
    target_leakage[t, j]  — fraction of period-j (entity|date)_cloze probe
                            answers that appear as a training-stream target of
                            an entity_cloze/date_cloze item in periods 0..t.
                            This is "direct answer leakage."

    context_leakage[t, j] — fraction of period-j probe answers that appear as
                            a substring in the input/evidence text of ANY
                            training item in periods 0..t.  Looser, captures
                            "the model has seen this answer in context."

Run:
    python scripts/analyze_probe_leakage.py
    python scripts/analyze_probe_leakage.py --output results/leakage/

Output
------
    results/leakage/leakage_target.csv     — matrix L_target[t, j]
    results/leakage/leakage_context.csv    — matrix L_context[t, j]
    results/leakage/leakage_summary.json   — by-period stats + readme
    stdout                                 — ASCII tables of both matrices
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Repo paths
_ROOT          = Path(__file__).resolve().parent.parent
_STREAM_DIR    = _ROOT / "local_data" / "cc_news" / "processed" / "stream_v2"
_PROBES_DIR    = _ROOT / "local_data" / "cc_news" / "processed" / "probes_v2"
_DEFAULT_PERIODS = ["2017_H1", "2017_H2", "2018_H1", "2018_H2"]

# Only these probe types are scored in the runner — they're what the regret
# matrix is computed over.
_SCORED_TYPES = {"entity_cloze", "date_cloze"}


# ── Text normalization ───────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Same normalization as runner's eval (lowercase, strip punctuation, collapse whitespace)."""
    return " ".join(
        s.lower().translate(str.maketrans("", "", string.punctuation)).split()
    )


# ── JSONL readers ────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path):
    """Yield JSON objects from a .jsonl file, skipping malformed lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_training_targets(period: str) -> Set[str]:
    """Normalized set of training-stream targets for entity_cloze/date_cloze
    items in this period.  These are the items whose target IS an answer (vs.
    completion/SSD where the target is a continuation/span and not directly
    comparable to a probe answer)."""
    path = _STREAM_DIR / f"{period}.jsonl"
    if not path.exists():
        return set()
    out: Set[str] = set()
    for obj in _iter_jsonl(path):
        task = obj.get("task", "")
        if task in _SCORED_TYPES:
            t = (obj.get("target") or "").strip()
            if t:
                out.add(_norm(t))
    return out


def load_training_context_blob(period: str) -> str:
    """One big lowercased blob of all training input + evidence text for this
    period.  Used for substring-match leakage check.  Memory cost: ~50 MB per
    period at n=200k items — fits comfortably in RAM."""
    path = _STREAM_DIR / f"{period}.jsonl"
    if not path.exists():
        return ""
    chunks: List[str] = []
    for obj in _iter_jsonl(path):
        inp = obj.get("input") or ""
        ev  = obj.get("evidence") or ""
        if inp: chunks.append(inp)
        if ev:  chunks.append(ev)
    return "\n".join(chunks).lower()


def load_probe_targets(period: str) -> List[str]:
    """Normalized list of (entity|date)_cloze probe targets for this period.
    Order preserved so we can count duplicates correctly."""
    path = _PROBES_DIR / f"{period}.jsonl"
    if not path.exists():
        return []
    out: List[str] = []
    for obj in _iter_jsonl(path):
        if obj.get("probe_type") in _SCORED_TYPES:
            t = (obj.get("target") or obj.get("answer") or "").strip()
            if t:
                out.append(_norm(t))
    return out


# ── Leakage computation ──────────────────────────────────────────────────────

def compute_leakage(periods: List[str]) -> Dict:
    """Compute target_leakage and context_leakage matrices."""
    n = len(periods)

    # 1) Build per-period sets / blobs (one pass each)
    print("Reading training streams …")
    train_targets:  Dict[str, Set[str]] = {}
    train_contexts: Dict[str, str]      = {}
    for p in periods:
        t0 = time.time()
        train_targets[p]  = load_training_targets(p)
        train_contexts[p] = load_training_context_blob(p)
        dt = time.time() - t0
        print(f"  {p}: {len(train_targets[p])} unique cloze targets, "
              f"{len(train_contexts[p])/1e6:.1f} MB context blob "
              f"({dt:.1f}s)")

    print("\nReading probes …")
    probe_targets: Dict[str, List[str]] = {}
    for p in periods:
        probe_targets[p] = load_probe_targets(p)
        print(f"  {p}: {len(probe_targets[p])} scored probes")

    # 2) Cumulative union of training targets across periods 0..t
    cum_targets: List[Set[str]] = []
    running: Set[str] = set()
    for p in periods:
        running = running | train_targets[p]
        cum_targets.append(set(running))

    # 3) Build leakage matrices
    print("\nComputing leakage matrices …")
    target_mat:  List[List[float]] = [[0.0] * n for _ in range(n)]
    context_mat: List[List[float]] = [[0.0] * n for _ in range(n)]

    for t in range(n):
        cumT  = cum_targets[t]
        # For context, OR together blob-substring checks; we precompute by
        # checking each probe target once and recording the earliest period
        # whose context contains it, then any t >= that period counts.
        for j in range(n):
            probes = probe_targets[periods[j]]
            if not probes:
                continue
            # Target leakage: count probes whose target ∈ cumT
            hit_t = sum(1 for p in probes if p in cumT)
            target_mat[t][j] = hit_t / len(probes)
            # Context leakage: count probes whose target is a substring of
            # any training-context blob in periods 0..t
            hit_c = 0
            for p_ans in probes:
                for ti in range(t + 1):
                    if p_ans and p_ans in train_contexts[periods[ti]]:
                        hit_c += 1
                        break
            context_mat[t][j] = hit_c / len(probes)
        print(f"  row t={t} ({periods[t]}) done")

    return {
        "periods": periods,
        "n_train_targets":  {p: len(train_targets[p])  for p in periods},
        "n_probes":         {p: len(probe_targets[p])  for p in periods},
        "target_leakage":   target_mat,
        "context_leakage":  context_mat,
    }


# ── Pretty-printing / IO ─────────────────────────────────────────────────────

def render_matrix(mat: List[List[float]], periods: List[str], title: str) -> str:
    n = len(periods)
    col_w = max(8, max(len(p) for p in periods) + 2)
    head = "after_period".ljust(col_w) + "".join(p.center(col_w) for p in periods)
    rule = "─" * len(head)
    lines = [f"{title}", rule, head, rule]
    for t in range(n):
        row = [periods[t].ljust(col_w)]
        for j in range(n):
            row.append(f"{mat[t][j]:.3f}".center(col_w))
        lines.append("".join(row))
    lines.append(rule)
    return "\n".join(lines)


def write_csv(mat: List[List[float]], periods: List[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("after_period," + ",".join(f"probes_{p}" for p in periods) + "\n")
        for t, pid in enumerate(periods):
            f.write(pid + "," + ",".join(f"{mat[t][j]:.4f}" for j in range(len(periods))) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="CC-News v2 probe-leakage analyzer")
    ap.add_argument("--periods", default=",".join(_DEFAULT_PERIODS),
                    help=f"Comma-separated periods (default: {','.join(_DEFAULT_PERIODS)})")
    ap.add_argument("--output", default=str(_ROOT / "results" / "leakage"),
                    help="Output directory (default: results/leakage)")
    args = ap.parse_args()

    periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = compute_leakage(periods)

    print()
    print(render_matrix(
        res["target_leakage"], periods,
        "Target leakage L[t,j] = frac. of period-j probe answers seen as a cloze training target in periods 0..t",
    ))
    print()
    print(render_matrix(
        res["context_leakage"], periods,
        "Context leakage L[t,j] = frac. of period-j probe answers found as substring of training text in periods 0..t",
    ))

    write_csv(res["target_leakage"],  periods, out_dir / "leakage_target.csv")
    write_csv(res["context_leakage"], periods, out_dir / "leakage_context.csv")
    (out_dir / "leakage_summary.json").write_text(
        json.dumps(res, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"Wrote {out_dir / 'leakage_target.csv'}")
    print(f"Wrote {out_dir / 'leakage_context.csv'}")
    print(f"Wrote {out_dir / 'leakage_summary.json'}")
    print()
    print("Interpretation:")
    print("  - High DIAGONAL means probes' answers are very common in their own period (expected).")
    print("  - High OFF-DIAGONAL L[t,j] with t>j means: training on later periods kept showing")
    print("    answers from earlier probes → no need to 'remember'  → expected flat BWT.")
    print("  - L[t,j] < 0.30 and the BWT row stays flat → real CL signal, not leakage.")


if __name__ == "__main__":
    sys.exit(main())
