"""CAPSEL Training UI — Backend Server.

REST + Server-Sent Events API for the training launcher web app.
Sends training commands to VS Code's integrated terminal via AppleScript (macOS)
or falls back to running as a subprocess whose output streams to the web UI.

Quick start
-----------
    pip install fastapi uvicorn pyyaml
    python app_server.py          # → http://localhost:7860

Or double-click  launch_ui.sh  (starts server + opens browser automatically).
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import platform
import re
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    import uvicorn
except ImportError:
    _missing.append("fastapi uvicorn")
try:
    from pydantic import BaseModel
except ImportError:
    _missing.append("pydantic")
if _missing:
    sys.exit(f"Missing packages — run:\n  pip install {' '.join(_missing)}\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR     = Path(__file__).resolve().parent
RESULTS_DIR = WORKDIR / "results"
VSCODE_DIR  = WORKDIR / ".vscode"
HTML_FILE   = WORKDIR / "training_launcher.html"

app = FastAPI(title="CAPSEL Training UI", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── In-memory run registry ────────────────────────────────────────────────────
_runs: Dict[str, dict] = {}   # run_id → run record


# ── Request models ────────────────────────────────────────────────────────────
class RunConfig(BaseModel):
    model_type:             str          = "inca"
    dataset:                str          = "cc_news"
    selector:               str          = "embedding_query"
    seed:                   int          = 42
    model_name:             str          = "google/flan-t5-base"
    batch_size:             int          = 32
    lr:                     str          = "3e-4"
    grad_accum_steps:       int          = 1
    epochs_per_period:      int          = 5
    max_input_length:       int          = 256
    k_eval:                 int          = 50
    patience:               int          = 5
    n_per_period:           int          = 20_000
    max_periods:            Optional[int] = None
    gradient_checkpointing: bool         = False
    use_adafactor:          bool         = False
    dry_run:                bool         = False
    out_dir:                str          = "results/"
    ablation:               Optional[str] = None
    baseline_mode:          str          = "sequential"
    device:                 str          = "auto"


# ── Command builder ───────────────────────────────────────────────────────────
def _build_command(cfg: RunConfig) -> str:
    if cfg.ablation:
        return (f"python scripts/run_ablation.py"
                f" --ablation {cfg.ablation} --dataset {cfg.dataset}")
    if cfg.model_type == "baseline":
        return (f"python scripts/train_baseline.py"
                f" --config configs/inca.yaml"
                f" --mode {cfg.baseline_mode}"
                f" --dataset {cfg.dataset}"
                f" --seed {cfg.seed}")
    parts = [
        "python scripts/train_inca.py",
        "--config configs/inca.yaml",
        f"--dataset {cfg.dataset}",
        f"--selector {cfg.selector}",
        f"--seed {cfg.seed}",
    ]
    if cfg.dry_run:          parts.append("--dry-run")
    if cfg.device != "auto": parts.append(f"--device {cfg.device}")
    return " ".join(parts)


# ── VS Code / shell integration ───────────────────────────────────────────────
def _write_vscode_task(label: str, command: str, log_file: Path) -> None:
    """Update .vscode/tasks.json so the run can be re-triggered from VS Code."""
    VSCODE_DIR.mkdir(exist_ok=True)
    tasks_path = VSCODE_DIR / "tasks.json"
    try:
        tasks = json.loads(tasks_path.read_text()) if tasks_path.exists() else {}
    except Exception:
        tasks = {}
    tasks.setdefault("version", "2.0.0")
    tasks.setdefault("tasks", [])
    tasks["tasks"] = [t for t in tasks["tasks"] if t.get("label") != label]
    tasks["tasks"].append({
        "label": label,
        "type": "shell",
        "command": f'{command} 2>&1 | tee "{log_file}"',
        "options": {"cwd": str(WORKDIR)},
        "group": {"kind": "build", "isDefault": False},
        "presentation": {"echo": True, "reveal": "always", "focus": True, "panel": "new"},
        "problemMatcher": [],
    })
    tasks_path.write_text(json.dumps(tasks, indent=2))


def _send_to_vscode_terminal(command: str, log_file: Path) -> bool:
    """
    macOS: use AppleScript to open VS Code's integrated terminal and run the
    command there. The command tees to log_file so the web UI can tail it.
    Returns True on success.
    """
    if platform.system() != "Darwin":
        return False

    tee_cmd = f'cd "{WORKDIR}" && {command} 2>&1 | tee "{log_file}"'
    # Escape double-quotes and backslashes for AppleScript
    tee_esc = tee_cmd.replace("\\", "\\\\").replace('"', '\\"')

    script = f'''
tell application "Visual Studio Code" to activate
delay 0.9
tell application "System Events"
    tell process "Code"
        -- Open / focus integrated terminal  (Ctrl-`)
        keystroke "`" using {{control down}}
        delay 0.7
        keystroke "{tee_esc}"
        key code 36
    end tell
end tell
'''
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, timeout=20,
    )
    return result.returncode == 0


def _run_subprocess_local(command: str, log_file: Path) -> int:
    """Fallback: run locally and capture output to log_file. Returns PID."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_file, "w", buffering=1)
    kwargs: dict = dict(
        shell=True, cwd=str(WORKDIR),
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    if platform.system() != "Windows":
        kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(command, **kwargs)
    return proc.pid


def _is_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ── Endpoints: health + run ───────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "workdir": str(WORKDIR),
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "results_dir": str(RESULTS_DIR),
    }


@app.post("/api/run")
def start_run(cfg: RunConfig):
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{uuid.uuid4().hex[:6]}"
    label  = f"CAPSEL: {cfg.dataset} {cfg.model_type} {cfg.selector} s{cfg.seed}"

    RESULTS_DIR.mkdir(exist_ok=True)
    log_file = RESULTS_DIR / f"run_{run_id}.log"
    log_file.touch()

    command = _build_command(cfg)
    _write_vscode_task(label, command, log_file)

    vscode_ok = _send_to_vscode_terminal(command, log_file)
    pid: Optional[int] = None
    if not vscode_ok:
        pid = _run_subprocess_local(command, log_file)

    run = {
        "run_id":          run_id,
        "status":          "running",
        "command":         command,
        "log_file":        str(log_file),
        "pid":             pid,
        "started_at":      datetime.now().isoformat(),
        "cfg":             cfg.dict(),
        "vscode_launched": vscode_ok,
        "label":           label,
    }
    _runs[run_id] = run
    return run


@app.post("/api/stop/{run_id}")
def stop_run(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    pid = run.get("pid")
    if pid and _is_alive(pid):
        try:
            if platform.system() != "Windows":
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    run["status"] = "stopped"
    return {"status": "stopped", "run_id": run_id}


@app.get("/api/runs")
def list_runs():
    for run in _runs.values():
        if run["status"] == "running" and run.get("pid"):
            if not _is_alive(run["pid"]):
                run["status"] = "completed"
    return list(reversed(list(_runs.values())))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    if run["status"] == "running" and run.get("pid"):
        if not _is_alive(run["pid"]):
            run["status"] = "completed"
    return run


# ── SSE log streaming ─────────────────────────────────────────────────────────
@app.get("/api/logs/{run_id}")
async def stream_logs(run_id: str, since: int = 0):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    log_path = Path(run["log_file"])

    async def generator() -> AsyncGenerator[str, None]:
        pos = since
        ticks_idle = 0
        while True:
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read(16384)
                    pos = f.tell()
                if chunk:
                    ticks_idle = 0
                    for line in chunk.splitlines():
                        payload = json.dumps({"line": line, "pos": pos})
                        yield f"data: {payload}\n\n"
                else:
                    ticks_idle += 1
                    # Check done
                    if run.get("pid") and not _is_alive(run["pid"]):
                        run["status"] = "completed"
                        yield f"data: {json.dumps({'event': 'done'})}\n\n"
                        return
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(0.25)
            except FileNotFoundError:
                await asyncio.sleep(1.0)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.get("/api/metrics/{run_id}")
def get_metrics(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    result: Dict[str, Any] = {
        "loss": [], "accuracy": [], "events": [], "n_blocks": 1,
    }

    # Parse log file for live metrics
    log_path = Path(run["log_file"])
    n_blocks = 1
    if log_path.exists():
        with open(log_path, "r", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                # [e0 s200] score=0.1234 cka=0.82 gnorm=0.54 loss=1.234 → NONE
                ms = re.search(r"\[e(\d+) s(\d+)\].*?loss=([\d.]+)", line)
                ma = re.search(r"score=([\d.]+)", line)
                if ms:
                    result["loss"].append({
                        "epoch": int(ms.group(1)),
                        "step":  int(ms.group(2)),
                        "loss":  float(ms.group(3)),
                    })
                if ma and ms:
                    result["accuracy"].append({
                        "step":  int(ms.group(2)),
                        "score": float(ma.group(1)),
                    })
                mb = re.search(r"Block chain:\s*(\d+)\s*block", line)
                if mb:
                    n_blocks = int(mb.group(1))
                if "BLOCK_FULL" in line or "PERIOD_LEARNED" in line:
                    result["events"].append({"msg": line, "ts": ""})

    result["n_blocks"] = n_blocks

    # Also check newest result sub-dir for loss_curve.csv (richer data)
    if RESULTS_DIR.exists():
        dirs = sorted(
            (d for d in RESULTS_DIR.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime, reverse=True,
        )
        for d in dirs[:3]:
            csv_path = d / "loss_curve.csv"
            if csv_path.exists():
                try:
                    with open(csv_path, newline="") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        result["loss_csv"] = [
                            {
                                "period": r.get("period", ""),
                                "block":  int(r.get("block", 0)),
                                "step":   int(r.get("opt_step", 0)),
                                "loss":   float(r.get("loss", 0)),
                            }
                            for r in rows
                        ]
                        break
                except Exception:
                    pass

    return result


# ── Results file browser ──────────────────────────────────────────────────────
@app.get("/api/results")
def list_results():
    if not RESULTS_DIR.exists():
        return []
    items = []
    for p in sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_dir():
            files = [f.name for f in p.iterdir()]
            items.append({
                "name":        p.name,
                "type":        "dir",
                "mtime":       p.stat().st_mtime,
                "files":       files,
                "has_loss":    "loss_curve.csv" in files,
                "checkpoints": [f for f in files if f.endswith(".pt")],
            })
        elif p.suffix == ".log":
            items.append({
                "name":  p.name,
                "type":  "log",
                "mtime": p.stat().st_mtime,
                "files": [p.name],
                "has_loss": False,
                "checkpoints": [],
            })
    return items[:60]


@app.get("/api/results/csv")
def read_csv_file(path: str):
    """Read a CSV inside results/ and return as list of dicts."""
    p = (RESULTS_DIR / path).resolve()
    try:
        p.relative_to(RESULTS_DIR)
    except ValueError:
        raise HTTPException(403, "Forbidden")
    if not p.exists():
        raise HTTPException(404, "Not found")
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


@app.get("/api/results/log")
def read_log_tail(path: str, lines: int = 200):
    """Return the last N lines of a log file."""
    p = (RESULTS_DIR / path).resolve()
    try:
        p.relative_to(RESULTS_DIR)
    except ValueError:
        raise HTTPException(403, "Forbidden")
    if not p.exists():
        raise HTTPException(404, "Not found")
    with open(p, "r", errors="replace") as f:
        all_lines = f.readlines()
    return {"lines": [l.rstrip() for l in all_lines[-lines:]]}


# ── Serve the SPA ─────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/ui")
def serve_ui():
    if HTML_FILE.exists():
        return FileResponse(HTML_FILE, media_type="text/html")
    return HTMLResponse(
        "<h1>training_launcher.html not found.</h1>"
        "<p>Make sure it is in the same directory as app_server.py</p>"
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'─'*56}")
    print(f"  CAPSEL Training UI  →  http://localhost:7860")
    print(f"  WorkDir : {WORKDIR}")
    print(f"  Results : {RESULTS_DIR}")
    print(f"  Platform: {platform.system()}")
    print(f"{'─'*56}\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
