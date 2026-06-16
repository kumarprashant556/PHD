# Paper B — End-to-End Runbook
## INCA vs LLaMA-Pro: Saturation-Driven Block Expansion on a Domain-Sequential Curriculum

> **What this file covers**: everything you need to go from a fresh clone to a
> completed sweep — environment, data, single runs, full orchestrated sweep,
> ablations, and result structure.
>
> **Comparison**: INCA (saturation-triggered) vs LLaMA-Pro / B6 (fixed-schedule)  
> **Dataset**: 3-domain sequential curriculum — Math → Code → Science  
> **Model**: FLAN-T5-base (250 M parameters)  
> **Hardware target**: single Apple M-series Mac (MPS) or 1× GPU (CUDA)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Data — What Downloads and From Where](#3-data--what-downloads-and-from-where)
4. [Single Run — INCA](#4-single-run--inca)
5. [Single Run — LLaMA-Pro Baseline (B6)](#5-single-run--llama-pro-baseline-b6)
6. [Full Sweep — Orchestrator (Recommended)](#6-full-sweep--orchestrator-recommended)
7. [Ablation Groups](#7-ablation-groups)
8. [Resume After Interruption](#8-resume-after-interruption)
9. [Output Directory Structure](#9-output-directory-structure)
10. [Reading Results](#10-reading-results)
11. [Common Errors and Fixes](#11-common-errors-and-fixes)

---

## 1. Prerequisites

| Requirement | Minimum | Tested |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB unified (M4) |
| Disk (models + data cache) | 15 GB | 20 GB |
| Internet | Required on first run | HuggingFace Hub |
| GPU / MPS | Optional (MPS auto-detected) | Apple M4 MPS |

HuggingFace will auto-download and cache:
- `google/flan-t5-base` — model weights (~1 GB)
- `lighteval/MATH` — math dataset (~100 MB)
- `bigcode/the-stack-smol` — code dataset (~5 GB)
- `allenai/sciq` — science dataset (~20 MB)

Set a custom cache location if needed:
```bash
export HF_HOME=/path/to/your/hf_cache   # default: ~/.cache/huggingface
export HF_DATASETS_CACHE=/path/to/your/datasets_cache
```

---

## 2. Environment Setup

This project uses the **`phd` conda environment**.

```bash
# Activate the environment (do this every terminal session)
conda activate phd

# Verify the install
python -c "import torch; print('torch:', torch.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"
```

If any package is missing, install into the phd env:
```bash
conda activate phd
pip install -r requirements.txt
```

**Check device detection** (tells you which backend training will use):
```bash
python -c "
import torch
if torch.cuda.is_available():
    print('Device: CUDA', torch.cuda.get_device_name(0))
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
    print('Device: Apple MPS')
else:
    print('Device: CPU (slow — consider GPU)')
"
```

> All scripts auto-detect the device. Override with `--device cpu|mps|cuda` if needed.

---

## 3. Data — What Downloads and From Where

All data is downloaded **automatically on first run** via HuggingFace `datasets`.
No manual download steps are required.

### 3.1 Training domains

| Period | Dataset | HF path | Size on disk | What is loaded |
|---|---|---|---|---|
| `P1_math` | lighteval/MATH | `lighteval/MATH` (config `all`) | ~100 MB | 12,500 problem+solution pairs; subsampled to `n_per_period=2000` |
| `P2_code` | The Stack Smol (Python) | `bigcode/the-stack-smol` | ~5 GB | Python files only; subsampled to 2000 |
| `P3_science` | SciQ | `allenai/sciq` | ~20 MB | 13,679 science passages; all splits merged; subsampled to 2000 |

### 3.2 Model weights

| Component | HF path | Size |
|---|---|---|
| FLAN-T5-base (encoder-decoder) | `google/flan-t5-base` | ~1 GB |

### 3.3 Completion framing

Every domain item is converted to a **text completion task** before training:

```
Input  →  "complete: <first 50% of text>"
Target →  "<remaining text, capped at 200 words>"
```

This is handled automatically by `data/domain_sequential.py` — no preprocessing script needed.

### 3.4 Pre-download (optional, for offline runs)

If you want to cache everything before going offline:
```bash
python -c "
from datasets import load_dataset
load_dataset('lighteval/MATH', 'all', split='train', trust_remote_code=True)
load_dataset('bigcode/the-stack-smol', data_dir='data/python', split='train', trust_remote_code=True)
load_dataset('allenai/sciq', split='train')
load_dataset('allenai/sciq', split='validation')
load_dataset('allenai/sciq', split='test')
print('All datasets cached.')
"

python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
AutoTokenizer.from_pretrained('google/flan-t5-base')
AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
print('Model cached.')
"
```

---

## 4. Single Run — INCA

Run INCA on the full 3-domain Paper B curriculum.

### 4.1 Default run (seed 42)

```bash
python scripts/train_inca.py \
    --config configs/paper_b.yaml
```

### 4.2 Override seed

```bash
python scripts/train_inca.py \
    --config configs/paper_b.yaml \
    --seed 123
```

### 4.3 Override device

```bash
# Force CPU (slower but universal)
python scripts/train_inca.py \
    --config configs/paper_b.yaml \
    --device cpu

# Force MPS (Apple Silicon)
python scripts/train_inca.py \
    --config configs/paper_b.yaml \
    --device mps

# Force CUDA
python scripts/train_inca.py \
    --config configs/paper_b.yaml \
    --device cuda
```

### 4.4 Override selector (for manual E-ROUTE testing)

```bash
# EmbeddingQuerySelector (default, S-QKV)
python scripts/train_inca.py --config configs/paper_b.yaml --selector embedding_query

# UCLBR (full three-component router)
python scripts/train_inca.py --config configs/paper_b.yaml --selector uclbr

# CrossAttentionSelector
python scripts/train_inca.py --config configs/paper_b.yaml --selector cross_attention

# WeightedSumSelector (control ablation)
python scripts/train_inca.py --config configs/paper_b.yaml --selector weighted_sum
```

### 4.5 Dry-run (verify config without training)

```bash
python scripts/train_inca.py \
    --config configs/paper_b.yaml \
    --dry-run
```

### 4.6 E-TIMING modes (manual)

```bash
# Saturation-triggered (INCA default — grow when signals fire)
python scripts/train_inca.py --config configs/paper_b.yaml --expand_at saturation

# Early (grow after 1 epoch regardless of signals)
python scripts/train_inca.py --config configs/paper_b.yaml --expand_at early

# Late (grow only after epochs exhausted)
python scripts/train_inca.py --config configs/paper_b.yaml --expand_at late

# Never (single fixed block — no growth)
python scripts/train_inca.py --config configs/paper_b.yaml --expand_at never
```

### 4.7 Expected console output

```
[INCA] Starting Paper B run — out_dir: results/paper_b/inca_v2_20260616_143022
[INCA] Device: mps | Precision: fp32
[INCA] Loading dataset: domain_sequential (n_per_period=2000, max_periods=3)
  [domain_sequential] P1_math: loading lighteval/MATH …
  P1_math: 2,000 examples (input ≈ 45 words, target ≈ 38 words)
  [domain_sequential] P2_code: loading bigcode/the-stack-smol Python …
  P2_code: 2,000 examples (input ≈ 120 words, target ≈ 95 words)
  [domain_sequential] P3_science: loading allenai/sciq …
  P3_science: 2,000 examples (input ≈ 60 words, target ≈ 40 words)

── Period 1/3: P1_math ────────────────────────────────────────
  [Trainer] Epoch 1/3 | step 62 | loss 3.214
  ...
  [Saturation] RIR=0.28 < 0.30, grad_decay=True, CKA=0.97 → BLOCK_FULL
  [INCA] Growing block 0→1 (params: 85M → 170M)
  Checkpoint saved: results/paper_b/inca_v2_.../inca_period_P1_math.pt
...
[INCA] Final checkpoint: results/paper_b/inca_v2_.../inca_v2_final.pt
[INCA] Memory log: results/paper_b/inca_v2_.../memory_log.json
```

---

## 5. Single Run — LLaMA-Pro Baseline (B6)

```bash
# Default run (seed 42)
python baselines/b6_llama_pro.py \
    --config configs/baselines/b6_paper_b.yaml \
    --initial_trainable_blocks 1

# Override seed
python baselines/b6_llama_pro.py \
    --config configs/baselines/b6_paper_b.yaml \
    --initial_trainable_blocks 1 \
    --seed 123

# Override device
python baselines/b6_llama_pro.py \
    --config configs/baselines/b6_paper_b.yaml \
    --initial_trainable_blocks 1 \
    --device mps
```

**What B6 does differently from INCA:**  
- Period 0: freezes all weights except the last encoder block  
- Period N > 0: deep-copies the last block, zeroes its output projections, appends it — regardless of whether the current block is saturated  
- No saturation signal; no selector; no replay buffer

---

## 6. Full Sweep — Orchestrator (Recommended)

The orchestrator at `scripts/run_paper_b.py` manages **81 jobs** across 6 groups,
tracks completion in a registry, and resumes interrupted runs automatically.

### 6.1 Check status (no training launched)

```bash
python scripts/run_paper_b.py --status
```

Sample output:
```
──────────────────────────────────────────────────────────────────────────
  Paper B sweep  ·  0/81 complete  ·  0 running  ·  0 failed
  Registry: results/paper_b/registry.json
──────────────────────────────────────────────────────────────────────────

  [main]  0/6
    ⬜  inca__main__seed42                        →
    ⬜  b6__main__seed42                          →
    ⬜  inca__main__seed123                       →
    ...
  [e_route]  0/12
  [e_sat]    0/27
  [e_cls3]   0/12
  [e_timing] 0/12
  [e_scope]  0/12
```

### 6.2 Run everything (picks up where it left off)

```bash
python scripts/run_paper_b.py
```

The orchestrator:
1. Loads the registry (creates it if this is the first run)
2. Syncs registry status with disk (auto-corrects stale `running` entries)
3. Skips `completed` and `failed` jobs
4. Resumes `running` jobs from their latest period checkpoint
5. Runs all `pending` jobs sequentially

### 6.3 Run a single group

```bash
# INCA default + LLaMA-Pro B6 (start here — 6 jobs)
python scripts/run_paper_b.py --group main

# Selector ablation (12 jobs)
python scripts/run_paper_b.py --group e_route

# Saturation threshold × patience sweep (27 jobs)
python scripts/run_paper_b.py --group e_sat

# Replay strategy ablation (12 jobs)
python scripts/run_paper_b.py --group e_cls3

# Expansion timing ablation — HEADLINE FIGURE (12 jobs)
python scripts/run_paper_b.py --group e_timing

# Lateral adapter rank ablation — appendix (12 jobs)
python scripts/run_paper_b.py --group e_scope
```

### 6.4 Run a single job by ID

```bash
# Format: inca__<group>__<suffix>  or  b6__main__seed<N>
python scripts/run_paper_b.py --job inca__main__seed42
python scripts/run_paper_b.py --job b6__main__seed42
python scripts/run_paper_b.py --job inca__e_timing__saturation__seed42
python scripts/run_paper_b.py --job inca__e_sat__rir0.3__pat3__seed42
python scripts/run_paper_b.py --job inca__e_scope__rank8__seed42
```

### 6.5 Preview without launching (dry-run)

```bash
python scripts/run_paper_b.py --dry-run
python scripts/run_paper_b.py --dry-run --group e_timing
```

### 6.6 Limit jobs per session (useful for timed runs)

```bash
# Run at most 3 jobs then stop
python scripts/run_paper_b.py --limit 3

# Run at most 5 jobs from the main group
python scripts/run_paper_b.py --group main --limit 5
```

### 6.7 Retry failed jobs

```bash
# Reset all failed → pending, then re-run
python scripts/run_paper_b.py --reset-failed
python scripts/run_paper_b.py
```

### 6.8 Recommended run order (for Mac M-series)

```bash
# Day 1: main results (6 jobs, ~2 h total)
python scripts/run_paper_b.py --group main

# Day 2: headline ablation (12 jobs, ~4 h)
python scripts/run_paper_b.py --group e_timing

# Day 3: remaining ablations
python scripts/run_paper_b.py --group e_route   # 12 jobs, ~4 h
python scripts/run_paper_b.py --group e_cls3    # 12 jobs, ~4 h

# Day 4: saturation sweep (largest group)
python scripts/run_paper_b.py --group e_sat     # 27 jobs, ~9 h

# Day 5: lateral adapter (appendix, can be skipped initially)
python scripts/run_paper_b.py --group e_scope   # 12 jobs, ~4 h
```

---

## 7. Ablation Groups

### 7.1 E-TIMING — expansion timing (12 jobs)
**The headline figure.** Tests when to grow vs the fixed-schedule of LLaMA-Pro.

| Job suffix | What happens | Expected rank |
|---|---|---|
| `early__seed*` | Grow after 1 epoch regardless | Worst |
| `saturation__seed*` | INCA default — grow when signals fire | Best |
| `late__seed*` | Grow only when epochs exhausted | 2nd |
| `never__seed*` | Single block — no growth at all | 3rd |

```bash
python scripts/run_paper_b.py --group e_timing
```

### 7.2 E-ROUTE — selector ablation (12 jobs)
Tests which routing mechanism best assigns inputs to blocks.

| Job suffix | Selector |
|---|---|
| `embedding_query__seed*` | S-QKV (recommended default) |
| `uclbr__seed*` | UCLBR (full router) |
| `cross_attention__seed*` | MLP gate on pooled outputs |
| `weighted_sum__seed*` | Input-independent scalar (control) |

```bash
python scripts/run_paper_b.py --group e_route
```

### 7.3 E-SAT — saturation threshold sweep (27 jobs)
Tests sensitivity of the saturation detector to its two key hyperparameters.

| Axis | Values |
|---|---|
| `rir_threshold` | 0.20, 0.30, 0.40 |
| `patience` | 3, 5, 8 |
| `seed` | 42, 123, 999 |

```bash
python scripts/run_paper_b.py --group e_sat
```

### 7.4 E-CLS3 — replay strategy (12 jobs)
Tests which replay sampling strategy best prevents forgetting.

| Job suffix | p_hard / p_easy / p_mid |
|---|---|
| `uniform__seed*` | 0.0 / 0.0 / 1.0 (uniform) |
| `hardest__seed*` | 1.0 / 0.0 / 0.0 (hard only) |
| `easiest__seed*` | 0.0 / 1.0 / 0.0 (easy only) |
| `schedule__seed*` | 0.7 / 0.2 / 0.1 (default) |

```bash
python scripts/run_paper_b.py --group e_cls3
```

### 7.5 E-SCOPE — lateral adapter rank (12 jobs, appendix)
Tests low-rank cross-block connections. `rank=0` matches main paper config.

| Job suffix | lateral_rank |
|---|---|
| `rank0__seed*` | 0 (no adapters — Phase 1 control) |
| `rank4__seed*` | 4 |
| `rank8__seed*` | 8 |
| `rank16__seed*` | 16 |

```bash
python scripts/run_paper_b.py --group e_scope
```

---

## 8. Resume After Interruption

### 8.1 Automatic resume (via orchestrator)

Just re-run the orchestrator — it handles everything:

```bash
python scripts/run_paper_b.py
```

The orchestrator:
- Detects jobs marked `running` in the registry (interrupted jobs)
- Scans the job's `out_dir` for `inca_period_<pid>.pt` checkpoint files
- Passes `--resume_dir` to the trainer if any checkpoint is found
- Restarts from scratch if no checkpoints exist (e.g., interrupted before P1 completed)

### 8.2 Manual resume (single job)

```bash
# Find the run directory (timestamped folder under results/paper_b/)
ls results/paper_b/inca_v2_*/run_id.json

# Resume directly
python scripts/train_inca.py \
    --config configs/paper_b.yaml \
    --resume_dir results/paper_b/inca_v2_20260616_143022
```

### 8.3 Resume granularity

| Interrupted at | Behaviour on resume |
|---|---|
| Mid-epoch in P1 | Restarts P1 from scratch (epoch-level loss is short) |
| Between P1 and P2 | Loads `inca_period_P1_math.pt`, starts from P2 |
| Mid-epoch in P2 | Restarts P2 from scratch, P1 already done |
| After P3 completes | Marks as completed, skips entirely |

---

## 9. Output Directory Structure

```
results/
└── paper_b/
    ├── registry.json                    ← central job registry (do not edit manually)
    │
    ├── inca_v2_<timestamp>/             ← one directory per INCA run
    │   ├── run_id.json                  ← identifies this run to the orchestrator
    │   ├── inca_period_P1_math.pt       ← period checkpoint (base_model + manager state)
    │   ├── inca_period_P2_code.pt
    │   ├── inca_period_P3_science.pt
    │   ├── inca_v2_final.pt             ← final checkpoint (written on clean completion)
    │   ├── memory_log.json              ← per-period: MEM_TRAIN, PAR, ACC/MB, wall_time
    │   └── run.log                      ← full training log
    │
    ├── b6/
    │   └── b6_llama_pro_<timestamp>/    ← one directory per B6 run
    │       ├── memory_log.json
    │       ├── metrics_summary.json     ← BWT, ACC, FWT, best period
    │       ├── regret_matrix.csv        ← R[t,j] = accuracy after period t on probes of j
    │       └── run.log
    │
    └── ablations/
        ├── e_timing/
        ├── e_route/
        ├── e_sat/
        ├── e_cls3/
        └── e_scope/
```

### 9.1 `registry.json` entry format

```json
{
  "inca__main__seed42": {
    "id": "inca__main__seed42",
    "method": "inca",
    "group": "main",
    "status": "completed",
    "out_dir": "results/paper_b/inca_v2_20260616_143022",
    "periods_done": ["P1_math", "P2_code", "P3_science"],
    "started_at": "2026-06-16T14:30:22",
    "completed_at": "2026-06-16T16:15:44",
    "error": null,
    "metrics": {}
  }
}
```

### 9.2 `memory_log.json` format

```json
{
  "method": "inca",
  "device": "mps",
  "periods": {
    "P1_math": {
      "peak_train_mb": 4821.3,
      "infer_mb": 2108.7,
      "param_total": 170234112,
      "param_trainable": 85117056,
      "param_delta": 85117056,
      "wall_time_s": 1842.1,
      "acc_delta": 0.312,
      "acc_per_mb": 0.000065
    },
    "P2_code": { ... },
    "P3_science": { ... }
  }
}
```

---

## 10. Reading Results

### 10.1 Quick status check

```bash
python scripts/run_paper_b.py --status
python scripts/run_paper_b.py --status --group main
```

### 10.2 Inspect a run's memory log

```bash
python -c "
import json, glob, sys
runs = sorted(glob.glob('results/paper_b/inca_v2_*/memory_log.json'))
if not runs:
    print('No completed INCA runs yet.')
    sys.exit(0)
latest = runs[-1]
print(f'Reading: {latest}')
data = json.load(open(latest))
print('\\nPeriod-level memory summary:')
for pid, row in data.get('periods', {}).items():
    print(f'  {pid}:  peak={row[\"peak_train_mb\"]:.0f} MB  '
          f'params={row[\"param_total\"]/1e6:.1f}M  '
          f'acc/MB={row[\"acc_per_mb\"]:.6f}  '
          f'time={row[\"wall_time_s\"]/60:.1f} min')
"
```

### 10.3 Compare INCA vs B6 memory

```bash
python -c "
import json, glob

inca_logs = sorted(glob.glob('results/paper_b/inca_v2_*/memory_log.json'))
b6_logs   = sorted(glob.glob('results/paper_b/b6/*/memory_log.json'))

for label, logs in [('INCA', inca_logs), ('B6', b6_logs)]:
    if not logs:
        print(f'{label}: no runs found'); continue
    data = json.load(open(logs[0]))
    periods = data.get('periods', {})
    total_mb  = sum(p['peak_train_mb'] for p in periods.values())
    total_par = max(p['param_total'] for p in periods.values())
    avg_acc   = sum(p.get('acc_delta', 0) for p in periods.values()) / max(len(periods), 1)
    print(f'{label}:  total_peak_MB={total_mb:.0f}  '
          f'final_params={total_par/1e6:.1f}M  avg_acc_delta={avg_acc:.3f}')
"
```

### 10.4 Check which jobs are done vs pending

```bash
python -c "
import json
data = json.load(open('results/paper_b/registry.json'))
from collections import Counter
by_status = Counter(v['status'] for v in data.values())
print('Status summary:', dict(by_status))
print()
failed = [k for k,v in data.items() if v['status'] == 'failed']
if failed:
    print('Failed jobs:')
    for j in failed:
        print(f'  {j}: {data[j][\"error\"]}')
"
```

---

## 11. Common Errors and Fixes

### `ModuleNotFoundError: No module named 'datasets'`
```bash
pip install -r requirements.txt
```

### `FileNotFoundError` for HuggingFace dataset
The dataset download failed or the cache is incomplete.
```bash
# Force re-download by clearing the cache entry
python -c "
from datasets import load_dataset
load_dataset('lighteval/MATH', 'all', split='train', trust_remote_code=True, download_mode='force_redownload')
"
```

### `RuntimeError: MPS backend out of memory`
Reduce batch size or use gradient checkpointing:
```bash
# Edit configs/paper_b.yaml:
#   batch_size: 4          (was 8)
#   grad_accum_steps: 8    (was 4; keeps effective batch = 32)
python scripts/train_inca.py --config configs/paper_b.yaml
```

### `RuntimeError: n_max_blocks reached`
The saturation detector grew too aggressively. Raise `n_max_blocks` or tighten thresholds:
```bash
# Temporary override via a quick YAML edit:
#   n_max_blocks: 12   (was 8)
#   rir_threshold: 0.15  (was 0.30 — harder to saturate)
```

### Registry shows job as `running` but no process is active
```bash
# Reset it to pending so it re-runs cleanly
python -c "
import json
data = json.load(open('results/paper_b/registry.json'))
data['inca__main__seed42']['status'] = 'pending'
import json; open('results/paper_b/registry.json','w').write(json.dumps(data, indent=2))
"
# Or reset all stale running jobs at once:
python scripts/run_paper_b.py --reset-failed
```

### `bigcode/the-stack-smol` download is very slow (~5 GB)
This is expected on first run. It caches to `~/.cache/huggingface/datasets/`.
Subsequent runs reuse the cache and load in seconds.

### HuggingFace rate limit / 403 error
Set your token if the dataset requires authentication:
```bash
huggingface-cli login
# or
export HUGGING_FACE_HUB_TOKEN=hf_...
```

---

## Config Quick Reference

| Config file | Purpose |
|---|---|
| `configs/paper_b.yaml` | INCA main Paper B config |
| `configs/baselines/b6_paper_b.yaml` | LLaMA-Pro B6 config (mirrors paper_b.yaml) |
| `configs/ablations/e_timing.yaml` | E-TIMING sweep definition |
| `configs/ablations/e_route.yaml` | E-ROUTE sweep definition |
| `configs/ablations/e_sat.yaml` | E-SAT sweep definition |
| `configs/ablations/e_cls3.yaml` | E-CLS3 sweep definition |
| `configs/ablations/e_scope.yaml` | E-SCOPE sweep definition |

Key hyperparameters in `configs/paper_b.yaml`:

```yaml
model_name:         "google/flan-t5-base"   # 250 M params
dataset:            "domain_sequential"      # math → code → science
n_per_period:       2000                     # items per domain
max_periods:        3                        # 3 domains
epochs_per_period:  3
batch_size:         8                        # effective batch = 32 (grad_accum × 4)
lr:                 3.0e-4
layers_per_block:   4                        # flan-t5-base: 12 enc layers → max 3 blocks
rir_threshold:      0.30                     # saturation signal sensitivity
patience:           3                        # plateau patience
lateral_rank:       0                        # 0 = Phase 1 (no adapters)
selector:           "embedding_query"        # S-QKV default
out_dir:            "results/paper_b"
```
