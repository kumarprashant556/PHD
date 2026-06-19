"""Paper 1 — single entry point.

Auto-detects the compute device and selects the right config:
  CUDA  →  configs/main_cuda.yaml  (bf16, batch 8, grad_accum 4)
  MPS   →  configs/main.yaml       (fp32, batch 4, grad_accum 8)
  CPU   →  configs/main.yaml       (slow — debug only)

Run from inside paper1/:
  python run.py                    # full sweep: INCA + B1–B7 ablations, 3 seeds
  python run.py --group main       # INCA default + B6 LLaMA-Pro only
  python run.py --status           # print job table, no training
  python run.py --group e_route    # single ablation group
  python run.py --job inca__main__seed42
  python run.py --reset-failed
  python run.py --from-scratch
  python run.py --from-scratch --group main
  python run.py --dry-run
  python run.py --limit 3

GPU selection (multi-GPU machines):
  CUDA_VISIBLE_DEVICES=0 python run.py

Job counts
----------
  main     :  3 INCA seeds + 3 B6 seeds          =  6
  e_route  :  4 selectors  × 3 seeds             = 12
  e_sat    :  3 rir_thresh × 3 patience × 3 seeds = 27
  e_cls3   :  4 replay strats × 3 seeds          = 12
  e_timing :  4 expand_at modes × 3 seeds        = 12
  e_scope  :  4 lateral ranks  × 3 seeds         = 12
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

# ── package root (paper1/ directory) ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── device auto-detection ─────────────────────────────────────────────────────
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

_DEVICE     = _detect_device()
_CFG_SUFFIX = "_cuda" if _DEVICE == "cuda" else ""   # "" → MPS/CPU config

# ── paths ─────────────────────────────────────────────────────────────────────
REGISTRY_PATH = _ROOT / "results" / "paper_b" / "registry.json"

# ─────────────────────────────────────────────────────────────────────────────
# Sweep definition
# ─────────────────────────────────────────────────────────────────────────────

def _inca_job(group: str, job_suffix: str, extra: dict) -> dict:
    return {
        "id":        f"inca__{group}__{job_suffix}",
        "method":    "inca",
        "group":     group,
        "config":    f"configs/main{_CFG_SUFFIX}.yaml",
        "overrides": extra,
    }


def _b6_job(seed: int) -> dict:
    return {
        "id":        f"b6__main__seed{seed}",
        "method":    "b6",
        "group":     "main",
        "config":    f"configs/baselines/b6_main{_CFG_SUFFIX}.yaml",
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

# ── Group: e_timing (expansion timing ablation — headline Fig 2) ──────────────
for _ea in ["early", "saturation", "late", "never"]:
    for _s in _SEEDS:
        SWEEP.append(_inca_job(
            "e_timing",
            f"{_ea}__seed{_s}",
            {"expand_at": _ea, "seed": _s},
        ))

# ── Group: e_scope (lateral adapter rank ablation) ───────────────────────────
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
        "status":       "pending",
        "out_dir":      None,
        "periods_done": [],
        "started_at":   None,
        "completed_at": None,
        "error":        None,
        "metrics":      {},
    }


def load_registry() -> Dict[str, dict]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    changed = False
    for job in SWEEP:
        if job["id"] not in data:
            data[job["id"]] = _default_entry(job)
            changed = True
    if changed:
        save_registry(data)
    return data


def save_registry(data: Dict[str, dict]) -> None:
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

    total   = sum(len(v) for v in groups.values())
    done    = sum(1 for e in data.values() if e["status"] == "completed"
                  and (not group or e["group"] == group))
    failed  = sum(1 for e in data.values() if e["status"] == "failed"
                  and (not group or e["group"] == group))
    running = sum(1 for e in data.values() if e["status"] == "running"
                  and (not group or e["group"] == group))

    print(f"\n{'─'*70}")
    print(f"  Paper 1 sweep  [{_DEVICE.upper()}]  ·  {done}/{total} complete  ·  "
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
# Disk helpers
# ─────────────────────────────────────────────────────────────────────────────

def _periods_done_on_disk(out_dir: str) -> List[str]:
    p = Path(out_dir)
    if not p.exists():
        return []
    return [ckpt.stem.replace("inca_period_", "")
            for ckpt in sorted(p.glob("inca_period_*.pt"))]


def _is_fully_done_on_disk(out_dir: str) -> bool:
    return (Path(out_dir) / "inca_v2_final.pt").exists()


def _b6_is_done(out_dir: str) -> bool:
    return (Path(out_dir) / "metrics_summary.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Job launchers
# ─────────────────────────────────────────────────────────────────────────────

def _build_inca_cmd_with_overrides(job: dict, resume_dir: Optional[str] = None):
    """Merge base config + overrides into a temp YAML, return (cmd, tmp_path)."""
    import yaml, copy

    base_path = _ROOT / job["config"]
    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    merged = copy.deepcopy(base_cfg)
    merged.update(job["overrides"])
    expand_at = merged.pop("expand_at", None)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="p1_job_",
        dir=_ROOT / "results" / "paper_b",
    )
    yaml.dump(merged, tmp)
    tmp.close()

    cmd = [sys.executable, str(_ROOT / "training" / "inca_trainer.py"), "--config", tmp.name]
    if expand_at:
        cmd += ["--expand_at", expand_at]
    if resume_dir:
        cmd += ["--resume_dir", resume_dir]
    return cmd, tmp.name


def _build_b6_cmd(job: dict):
    """Merge base config + overrides into a temp YAML, return (cmd, tmp_path)."""
    import yaml, copy

    base_path = _ROOT / job["config"]
    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    merged = copy.deepcopy(base_cfg)
    merged.update(job["overrides"])

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="p1_b6_",
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
    job_id = job["id"]

    resume_dir = None
    if entry["status"] == "running" and entry["out_dir"]:
        periods_done = _periods_done_on_disk(entry["out_dir"])
        if periods_done:
            resume_dir = entry["out_dir"]
            print(f"  ↺  Resuming from {Path(resume_dir).name}  "
                  f"(periods done: {periods_done})")
        else:
            print(f"  ↺  Restarting {job_id} (no period checkpoints found)")

    if dry_run:
        print(f"  [dry-run] {'RESUME' if resume_dir else 'RUN'} {job_id}")
        return True

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
            if entry.get("out_dir") and Path(entry["out_dir"]).exists():
                out_dir_found = entry["out_dir"]
            else:
                pb_dir = _ROOT / "results" / "paper_b"
                candidates = sorted(
                    pb_dir.glob("inca_v2_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                out_dir_found = next(
                    (str(d) for d in candidates if (d / "run_id.json").exists()),
                    str(candidates[0]) if candidates else "",
                )

            periods_done = _periods_done_on_disk(out_dir_found) if job["method"] == "inca" else []
            _update(data, job_id,
                    status="completed", out_dir=out_dir_found,
                    periods_done=periods_done,
                    completed_at=datetime.now().isoformat(), error=None)
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


def _sync_registry_with_disk(data: Dict[str, dict]) -> None:
    changed = False
    for entry in data.values():
        if entry["status"] not in ("running", "completed") or not entry["out_dir"]:
            continue
        od = entry["out_dir"]
        if entry["method"] == "inca":
            if _is_fully_done_on_disk(od) and entry["status"] != "completed":
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


def _clean_jobs(data: Dict[str, dict], affected_ids: List[str]) -> None:
    import shutil

    pb_dir  = REGISTRY_PATH.parent
    cleaned: set = set()

    for jid in affected_ids:
        od = data[jid].get("out_dir")
        if od:
            p = Path(od)
            if p.exists() and str(p) not in cleaned:
                shutil.rmtree(p, ignore_errors=True)
                cleaned.add(str(p))
                print(f"    deleted  {p.relative_to(_ROOT)}")

    affected_methods = {data[jid]["method"] for jid in affected_ids}
    if "inca" in affected_methods:
        for d in sorted(pb_dir.glob("inca_v2_*")):
            if str(d) not in cleaned:
                shutil.rmtree(d, ignore_errors=True)
                cleaned.add(str(d))
                print(f"    deleted  {d.relative_to(_ROOT)}")

    if "b6" in affected_methods:
        b6_dir = pb_dir / "b6"
        if b6_dir.exists():
            for d in sorted(b6_dir.iterdir()):
                if d.is_dir() and str(d) not in cleaned:
                    shutil.rmtree(d, ignore_errors=True)
                    cleaned.add(str(d))
                    print(f"    deleted  b6/{d.name}")

    for tmp in pb_dir.glob("p1_job_*.yaml"):
        tmp.unlink(missing_ok=True)
    for tmp in pb_dir.glob("p1_b6_*.yaml"):
        tmp.unlink(missing_ok=True)

    for jid in affected_ids:
        data[jid].update({
            "status": "pending", "out_dir": None, "periods_done": [],
            "started_at": None, "completed_at": None, "error": None, "metrics": {},
        })

    save_registry(data)
    print(f"\n  ✅  Removed {len(cleaned)} output dir(s).  "
          f"{len(affected_ids)} job(s) reset to pending.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Paper 1 sweep orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--status",       action="store_true")
    p.add_argument("--group",        default=None,
                   choices=["main", "e_route", "e_sat", "e_cls3", "e_timing", "e_scope"])
    p.add_argument("--job",          default=None)
    p.add_argument("--reset-failed", action="store_true")
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--limit",        type=int, default=None)
    args = p.parse_args()

    print(f"Device: {_DEVICE.upper()}  →  config suffix: '{_CFG_SUFFIX or 'none (MPS/CPU)'}'")

    data = load_registry()
    _sync_registry_with_disk(data)

    if args.from_scratch:
        affected_ids = (
            [args.job] if args.job and args.job in data
            else [j["id"] for j in SWEEP if not args.group or j["group"] == args.group]
        )
        scope = (f"group={args.group}" if args.group
                 else f"job={args.job}" if args.job else "ALL groups")
        print(f"\n🗑  --from-scratch  ({scope})  —  cleaning {len(affected_ids)} job(s) …")
        _clean_jobs(data, affected_ids)
        data = load_registry()

    if args.reset_failed:
        n = sum(1 for e in data.values() if e["status"] == "failed")
        for entry in data.values():
            if entry["status"] == "failed":
                entry["status"] = "pending"
                entry["error"]  = None
        save_registry(data)
        print(f"Reset {n} failed job(s) to pending.")

    print_status(data, group=args.group)

    if args.status:
        return

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

    launched = 0
    for job in queue:
        if args.limit and launched >= args.limit:
            print(f"\nLimit of {args.limit} job(s) reached.  Re-run to continue.\n")
            break
        ok = run_job(job, data[job["id"]], data, dry_run=args.dry_run)
        if ok:
            launched += 1

    data = load_registry()
    print_status(data, group=args.group)


if __name__ == "__main__":
    main()
