"""Paper B sweep orchestrator with checkpoint-aware resume.

Tracks every INCA-variant run (main + E-ROUTE + E-SAT + E-CLS3 + E-TIMING +
E-SCOPE) and the LLaMA-Pro B6 baseline in a central registry.  On each
invocation it prints the current status table, then runs whatever is pending
or resumable.

Comparison scope: INCA vs LLaMA-Pro (B6) only.

Commands
--------
  # Show status and run all pending jobs sequentially
  python scripts/run_paper_b.py

  # Status only (no training)
  python scripts/run_paper_b.py --status

  # Run only a specific group
  python scripts/run_paper_b.py --group main
  python scripts/run_paper_b.py --group e_route
  python scripts/run_paper_b.py --group e_sat
  python scripts/run_paper_b.py --group e_cls3
  python scripts/run_paper_b.py --group e_timing
  python scripts/run_paper_b.py --group e_scope

  # Run a single job by ID (useful for debugging)
  python scripts/run_paper_b.py --job inca__main__seed42

  # Mark all failed jobs as pending so they re-run
  python scripts/run_paper_b.py --reset-failed

  # Preview what would run without launching anything
  python scripts/run_paper_b.py --dry-run

  # Limit the number of jobs to launch this session
  python scripts/run_paper_b.py --limit 3

Registry
--------
  results/paper_b/registry.json   — one entry per job, updated atomically

Resume behaviour
----------------
- completed → skipped
- failed    → skipped (re-run with --reset-failed or --job JOB_ID)
- running   → treated as interrupted; resume from latest period checkpoint
- pending   → run from scratch

Job counts
----------
  main     :  3 INCA seeds + 3 B6 seeds          =  6
  e_route  :  4 selectors  × 3 seeds             = 12
  e_sat    :  3 rir_thresh × 3 patience × 3 seeds = 27
  e_cls3   :  4 replay strats × 3 seeds          = 12
  e_timing :  4 expand_at modes × 3 seeds        = 12   ← Figure 2 (headline)
  e_scope  :  4 lateral ranks  × 3 seeds         = 12   ← appendix
  ─────────────────────────────────────────────────────
  Total    :                                       81
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── repo root on sys.path so `from training...` resolves ─────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REGISTRY_PATH = _ROOT / "results" / "paper_b" / "registry.json"

# ─────────────────────────────────────────────────────────────────────────────
# Sweep definition
#
# Each entry: {id, method, group, config, overrides}
#   id       : unique stable string key (used in registry + CLI)
#   method   : "inca" | "b6"  (maps to the launcher below)
#   group    : "main" | "e_route" | "e_sat" | "e_cls3" | "e_timing" | "e_scope"
#   config   : path to YAML config (relative to repo root)
#   overrides: dict of CLI --key value overrides applied on top of the config
# ─────────────────────────────────────────────────────────────────────────────

def _inca_job(group: str, job_suffix: str, extra: dict) -> dict:
    return {
        "id":        f"inca__{group}__{job_suffix}",
        "method":    "inca",
        "group":     group,
        "config":    "configs/paper_b.yaml",
        "overrides": extra,
    }


def _b6_job(seed: int) -> dict:
    return {
        "id":        f"b6__main__seed{seed}",
        "method":    "b6",
        "group":     "main",
        "config":    "configs/baselines/b6_paper_b.yaml",
        "overrides": {"seed": seed},
    }


_SEEDS = [42, 123, 999]

SWEEP: List[Dict[str, Any]] = []

# ── Group: main (INCA default + LLaMA-Pro B6) ────────────────────────────────
for _s in _SEEDS:
    SWEEP.append(_inca_job("main", f"seed{_s}", {"seed": _s}))
    SWEEP.append(_b6_job(_s))

# ── Group: e_route (selector ablation) ───────────────────────────────────────
for _sel in ["embedding_query", "uclbr", "cross_attention", "weighted_sum"]:
    for _s in _SEEDS:
        SWEEP.append(_inca_job(
            "e_route",
            f"{_sel}__seed{_s}",
            {"selector": _sel, "seed": _s},
        ))

# ── Group: e_sat (saturation threshold × patience ablation) ──────────────────
for _rir in [0.20, 0.30, 0.40]:
    for _pat in [3, 5, 8]:
        for _s in _SEEDS:
            SWEEP.append(_inca_job(
                "e_sat",
                f"rir{_rir}__pat{_pat}__seed{_s}",
                {"rir_threshold": _rir, "patience": _pat, "seed": _s},
            ))

# ── Group: e_cls3 (replay strategy ablation) ─────────────────────────────────
_REPLAY_OVERRIDES = {
    "uniform":  {"p_hard": 0.0,  "p_easy": 0.0,  "p_mid": 1.0},
    "hardest":  {"p_hard": 1.0,  "p_easy": 0.0,  "p_mid": 0.0},
    "easiest":  {"p_hard": 0.0,  "p_easy": 1.0,  "p_mid": 0.0},
    "schedule": {"p_hard": 0.70, "p_easy": 0.20, "p_mid": 0.10},
}
for _strat, _rep_ov in _REPLAY_OVERRIDES.items():
    for _s in _SEEDS:
        SWEEP.append(_inca_job(
            "e_cls3",
            f"{_strat}__seed{_s}",
            {**_rep_ov, "seed": _s},
        ))

# ── Group: e_timing (expansion timing ablation) ───────────────────────────────
# THE headline figure: saturation-triggered vs fixed-schedule expansion.
# expand_at is a CLI flag (--expand_at), not a YAML key; the orchestrator
# pops it from overrides and passes it to train_inca.py separately.
# Expected ordering: saturation >= late > never >> early
for _ea in ["early", "saturation", "late", "never"]:
    for _s in _SEEDS:
        SWEEP.append(_inca_job(
            "e_timing",
            f"{_ea}__seed{_s}",
            {"expand_at": _ea, "seed": _s},
        ))

# ── Group: e_scope (lateral adapter rank ablation — Phase 2 / appendix) ──────
# lateral_rank=0 matches the main Paper B config (Phase 1, no adapters).
# lateral_rank > 0 activates LateralAdapter in layer_manager.py.
# If any rank > 0 shows >= 1% improvement, promote to main table.
for _rank in [0, 4, 8, 16]:
    for _s in _SEEDS:
        SWEEP.append(_inca_job(
            "e_scope",
            f"rank{_rank}__seed{_s}",
            {"lateral_rank": _rank, "seed": _s},
        ))

# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_entry(job: dict) -> dict:
    return {
        "id":           job["id"],
        "method":       job["method"],
        "group":        job["group"],
        "config":       job["config"],
        "overrides":    job["overrides"],
        "status":       "pending",     # pending | running | completed | failed
        "out_dir":      None,
        "periods_done": [],
        "started_at":   None,
        "completed_at": None,
        "error":        None,
        "metrics":      {},
    }


def load_registry() -> Dict[str, dict]:
    """Load registry from disk; back-fill any jobs that aren't in it yet."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    # Ensure every SWEEP job has an entry
    changed = False
    for job in SWEEP:
        if job["id"] not in data:
            data[job["id"]] = _default_entry(job)
            changed = True
    if changed:
        save_registry(data)
    return data


def save_registry(data: Dict[str, dict]) -> None:
    """Write registry atomically (temp-file + rename)."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(REGISTRY_PATH)


def _update(data: Dict[str, dict], job_id: str, **kwargs) -> None:
    data[job_id].update(kwargs)
    save_registry(data)


# ─────────────────────────────────────────────────────────────────────────────
# Status display
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {
    "pending":   "⬜",
    "running":   "🔄",
    "completed": "✅",
    "failed":    "❌",
}


def print_status(data: Dict[str, dict], group: Optional[str] = None) -> None:
    groups: Dict[str, List[dict]] = {}
    for entry in data.values():
        g = entry["group"]
        if group and g != group:
            continue
        groups.setdefault(g, []).append(entry)

    total = sum(len(v) for v in groups.values())
    done  = sum(1 for e in data.values() if e["status"] == "completed"
                and (not group or e["group"] == group))
    failed = sum(1 for e in data.values() if e["status"] == "failed"
                 and (not group or e["group"] == group))
    running = sum(1 for e in data.values() if e["status"] == "running"
                  and (not group or e["group"] == group))

    print(f"\n{'─'*70}")
    print(f"  Paper B sweep  ·  {done}/{total} complete  ·  "
          f"{running} running  ·  {failed} failed")
    print(f"  Registry: {REGISTRY_PATH}")
    print(f"{'─'*70}")

    for g_name, entries in sorted(groups.items()):
        g_done  = sum(1 for e in entries if e["status"] == "completed")
        g_total = len(entries)
        print(f"\n  [{g_name}]  {g_done}/{g_total}")
        for e in entries:
            icon    = _STATUS_ICON.get(e["status"], "?")
            periods = f"periods={e['periods_done']}" if e["periods_done"] else ""
            out     = f"→ {Path(e['out_dir']).name}" if e["out_dir"] else ""
            err     = f"  ERR: {e['error'][:60]}" if e["error"] else ""
            print(f"    {icon}  {e['id']:<50}  {periods}  {out}{err}")

    print(f"\n{'─'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Detect completed periods from disk (without running the trainer)
# ─────────────────────────────────────────────────────────────────────────────

def _periods_done_on_disk(out_dir: str) -> List[str]:
    """Scan an existing INCA run directory for completed period checkpoints."""
    p = Path(out_dir)
    if not p.exists():
        return []
    done = []
    for ckpt in sorted(p.glob("inca_period_*.pt")):
        pid = ckpt.stem.replace("inca_period_", "")
        done.append(pid)
    return done


def _is_fully_done_on_disk(out_dir: str) -> bool:
    """True if the run directory contains the final INCA checkpoint."""
    return (Path(out_dir) / "inca_v2_final.pt").exists()


def _b6_is_done(out_dir: str) -> bool:
    """True if the B6 run directory contains the metrics summary."""
    return (Path(out_dir) / "metrics_summary.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Job launchers
# ─────────────────────────────────────────────────────────────────────────────

def _build_inca_cmd(job: dict, resume_dir: Optional[str] = None) -> List[str]:
    """Build the subprocess command for an INCA job."""
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "train_inca.py"),
        "--config", str(_ROOT / job["config"]),
    ]
    ov = job["overrides"]
    if "seed"          in ov: cmd += ["--seed",     str(ov["seed"])]
    if "selector"      in ov: cmd += ["--selector", str(ov["selector"])]
    # Numeric overrides not exposed as CLI flags need a temporary YAML or
    # direct cfg_dict override.  For now, supported flags: seed, selector.
    # Complex overrides (rir_threshold, patience, p_hard, …) are injected
    # via a temporary override YAML generated below.
    if resume_dir:
        cmd += ["--resume_dir", resume_dir]
    return cmd


def _build_inca_cmd_with_overrides(job: dict, resume_dir: Optional[str] = None) -> List[str]:
    """Write a temporary override YAML and pass it as the config.

    This handles all numeric/non-flag overrides cleanly without modifying
    train_inca.py's argument parser for every possible key.

    Special handling
    ----------------
    ``expand_at`` — a CLI-only flag (--expand_at) that inca_trainer.py
      translates into cfg overrides at runtime.  It is NOT a valid INCAConfig
      field, so it must be popped from the merged dict before writing the YAML
      and passed as --expand_at instead.

    ``lateral_rank`` — a genuine INCAConfig field; goes into the YAML as-is
      and is read by INCALayerManager to activate LateralAdapter (E-SCOPE).
    """
    import yaml, tempfile, copy

    # Load base config
    base_path = _ROOT / job["config"]
    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    # Apply overrides
    merged = copy.deepcopy(base_cfg)
    merged.update(job["overrides"])

    # Pop CLI-only keys before writing YAML
    expand_at = merged.pop("expand_at", None)

    # Write merged config to a temp file that persists for the subprocess lifetime
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="paper_b_job_",
        dir=_ROOT / "results" / "paper_b",
    )
    yaml.dump(merged, tmp)
    tmp.close()

    cmd = [
        sys.executable, str(_ROOT / "scripts" / "train_inca.py"),
        "--config", tmp.name,
    ]
    if expand_at:
        cmd += ["--expand_at", expand_at]
    if resume_dir:
        cmd += ["--resume_dir", resume_dir]
    return cmd, tmp.name   # caller should delete tmp after subprocess completes


def _build_b6_cmd(job: dict) -> List[str]:
    """Build the subprocess command for a B6 (LLaMA-Pro) job."""
    # B6 doesn't support resume yet; it restarts if interrupted.
    # (Period times are short enough on Paper B data that this is acceptable.)
    import yaml, tempfile, copy
    base_path = _ROOT / job["config"]
    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    merged = copy.deepcopy(base_cfg)
    merged.update(job["overrides"])
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="paper_b_b6_",
        dir=_ROOT / "results" / "paper_b",
    )
    yaml.dump(merged, tmp)
    tmp.close()
    cmd = [
        sys.executable, str(_ROOT / "baselines" / "b6_llama_pro.py"),
        "--config", tmp.name,
        "--initial_trainable_blocks", "1",
    ]
    return cmd, tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Main run loop
# ─────────────────────────────────────────────────────────────────────────────

def run_job(job: dict, entry: dict, data: Dict[str, dict], dry_run: bool) -> bool:
    """Run a single job.  Returns True if successful, False if failed/skipped."""
    job_id = job["id"]

    # Determine if we're resuming (running state with an existing out_dir)
    resume_dir = None
    if entry["status"] == "running" and entry["out_dir"]:
        # Process was interrupted — check what's already done on disk
        periods_done = _periods_done_on_disk(entry["out_dir"])
        if periods_done:
            resume_dir = entry["out_dir"]
            print(f"  ↺  Resuming from {Path(resume_dir).name}  "
                  f"(periods done: {periods_done})")
        else:
            print(f"  ↺  Restarting {job_id} (no period checkpoints found in "
                  f"{Path(entry['out_dir']).name})")

    if dry_run:
        action = "RESUME" if resume_dir else "RUN"
        print(f"  [dry-run] {action} {job_id}")
        return True

    # Mark as running
    _update(data, job_id, status="running", started_at=datetime.now().isoformat())

    tmp_yaml = None
    try:
        if job["method"] == "inca":
            cmd, tmp_yaml = _build_inca_cmd_with_overrides(job, resume_dir)
        elif job["method"] == "b6":
            cmd, tmp_yaml = _build_b6_cmd(job)
        else:
            raise ValueError(f"Unknown method: {job['method']}")

        print(f"\n  ▶  {job_id}")
        print(f"     {' '.join(cmd[:6])} …")

        env = {**os.environ, "PYTHONPATH": str(_ROOT)}
        result = subprocess.run(cmd, env=env, cwd=str(_ROOT))

        if result.returncode == 0:
            # Detect the actual out_dir from disk.
            # Priority: (1) the entry already has an out_dir (resume case),
            # (2) find the newest inca_v2_* dir that contains a run_id.json.
            if entry.get("out_dir") and Path(entry["out_dir"]).exists():
                out_dir_found = entry["out_dir"]
            else:
                pb_dir = _ROOT / "results" / "paper_b"
                candidate_dirs = sorted(
                    pb_dir.glob("inca_v2_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                out_dir_found = next(
                    (str(d) for d in candidate_dirs if (d / "run_id.json").exists()),
                    str(candidate_dirs[0]) if candidate_dirs else "",
                )

            periods_done = _periods_done_on_disk(out_dir_found) if job["method"] == "inca" else []
            _update(
                data, job_id,
                status="completed",
                out_dir=out_dir_found,
                periods_done=periods_done,
                completed_at=datetime.now().isoformat(),
                error=None,
            )
            print(f"  ✅  {job_id}")
            return True
        else:
            err_msg = f"exit code {result.returncode}"
            _update(data, job_id, status="failed", error=err_msg)
            print(f"  ❌  {job_id}  ({err_msg})")
            return False

    except Exception as exc:
        _update(data, job_id, status="failed", error=str(exc))
        print(f"  ❌  {job_id}  ({exc})")
        return False
    finally:
        if tmp_yaml and Path(tmp_yaml).exists():
            try:
                os.unlink(tmp_yaml)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Out-dir tracking: write a tiny marker file so the orchestrator can find
# the run dir even when the trainer doesn't print it cleanly.
# ─────────────────────────────────────────────────────────────────────────────

def _sync_registry_with_disk(data: Dict[str, dict]) -> None:
    """Update 'periods_done' and status for 'running' entries by scanning disk."""
    changed = False
    for entry in data.values():
        if entry["status"] not in ("running", "completed"):
            continue
        if not entry["out_dir"]:
            continue
        od = entry["out_dir"]
        if entry["method"] == "inca":
            if _is_fully_done_on_disk(od):
                if entry["status"] != "completed":
                    entry["status"] = "completed"
                    changed = True
            periods = _periods_done_on_disk(od)
            if periods != entry.get("periods_done"):
                entry["periods_done"] = periods
                changed = True
        elif entry["method"] == "b6":
            if _b6_is_done(od) and entry["status"] != "completed":
                entry["status"] = "completed"
                changed = True
    if changed:
        save_registry(data)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Paper B sweep orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--status",       action="store_true",
                   help="Print status table and exit")
    p.add_argument("--group",        default=None,
                   choices=["main", "e_route", "e_sat", "e_cls3", "e_timing", "e_scope"],
                   help="Run only this ablation group")
    p.add_argument("--job",          default=None,
                   help="Run a single job by ID")
    p.add_argument("--reset-failed", action="store_true",
                   help="Mark all failed jobs as pending so they re-run")
    p.add_argument("--dry-run",      action="store_true",
                   help="Show what would run without launching anything")
    p.add_argument("--limit",        type=int, default=None,
                   help="Launch at most N jobs this session")
    args = p.parse_args()

    # ── load + sync registry ───────────────────────────────────────────
    data = load_registry()
    _sync_registry_with_disk(data)

    # ── --reset-failed ─────────────────────────────────────────────────
    if args.reset_failed:
        n = 0
        for entry in data.values():
            if entry["status"] == "failed":
                entry["status"] = "pending"
                entry["error"]  = None
                n += 1
        save_registry(data)
        print(f"Reset {n} failed job(s) to pending.")

    # ── status table ───────────────────────────────────────────────────
    print_status(data, group=args.group)

    if args.status:
        return

    # ── build job queue ────────────────────────────────────────────────
    if args.job:
        if args.job not in data:
            print(f"Unknown job '{args.job}'.  Valid IDs:")
            for jid in sorted(data):
                print(f"  {jid}")
            sys.exit(1)
        queue = [j for j in SWEEP if j["id"] == args.job]
    else:
        queue = [
            j for j in SWEEP
            if (not args.group or j["group"] == args.group)
            and data[j["id"]]["status"] in ("pending", "running")
        ]

    if not queue:
        print("Nothing to run.  All jobs are completed or failed.")
        print("Use --reset-failed to retry failed jobs.\n")
        return

    print(f"Jobs to run: {len(queue)}"
          + (f"  (limit: {args.limit})" if args.limit else ""))

    # ── run queue ──────────────────────────────────────────────────────
    launched = 0
    for job in queue:
        if args.limit and launched >= args.limit:
            print(f"\nLimit of {args.limit} job(s) reached.  "
                  "Re-run to continue remaining jobs.")
            break
        entry = data[job["id"]]
        ok = run_job(job, entry, data, dry_run=args.dry_run)
        if ok:
            launched += 1

    # ── final status ───────────────────────────────────────────────────
    data = load_registry()
    print_status(data, group=args.group)


if __name__ == "__main__":
    main()
