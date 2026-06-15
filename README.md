# CAPSEL — Continual Adaptive Plasticity via Saturation-Triggered Expansion and Learning

> A continually-growing FLAN-T5 seq2seq model for temporal language modelling.  
> **INCA** (**I**ncremental **N**eural **C**hain **A**rchitecture) dynamically grows new transformer blocks when a multi-signal saturation detector decides the current capacity is exhausted for a given time period.

---

## Key Idea

Standard continual learning fine-tunes a fixed-capacity model, leading to catastrophic forgetting. INCA instead **freezes** saturated blocks and **grows** new ones — building a chain of specialised, temporally-aware encoders with no forgetting by construction.

```
Period 1          Period 2          Period 3
[Block 0]  →  [Block 0 frozen]  →  [Block 0 frozen]
              [Block 1]         →  [Block 1 frozen]
                                   [Block 2]
```

The selector routes each input to the most relevant block via embedding-query attention (or UCLBR for load-balanced routing).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Training UI  →  http://localhost:7860
bash launch_ui.sh

# 3. Or train directly from CLI
python scripts/train_inca.py \
    --config configs/inca.yaml \
    --dataset cc_news \
    --selector embedding_query \
    --seed 42

# Baseline (shows forgetting)
python scripts/train_baseline.py \
    --config configs/inca.yaml \
    --mode sequential \
    --dataset cc_news \
    --seed 42
```

---

## Repository Layout

```
WorkingDir/
├── app_server.py            ← FastAPI backend (REST + SSE)  →  :7860
├── training_launcher.html   ← SPA frontend (5 pages)
├── launch_ui.sh             ← One-command launcher
│
├── configs/
│   ├── inca.yaml            ← Main hyperparameter config (maps 1-to-1 to INCAConfig)
│   ├── base.yaml            ← Shared base config
│   └── ablations/           ← Per-ablation YAML overrides
│
├── data/                    ← Dataset loaders — all return Dict[period_id, Dataset]
│   │                          columns: input_text · target_text · period
│   ├── _base.py             ← Abstract base + shared finalise() / completion-framing pipeline
│   ├── tokenizer.py         ← build_tokenized_periods, make_dataloader, replay mixing
│   ├── cc_news.py           ← CC-News         (4 half-year periods, 2017–2018)  [local]
│   ├── streaming_qa.py      ← StreamingQA     (26 monthly periods, 2017–2019)  [local]
│   ├── temporalwiki.py      ← TemporalWiki    (2 Wikipedia snapshots)           [local]
│   ├── tic_lm.py            ← TiC-LM          (9 daily C4 slices, Apr 2019)    [local]
│   ├── redpajama.py         ← RedPajama-V2    (HF streaming)
│   └── __init__.py          ← load_periods() dispatcher
│
├── datasets/                ← Local processed data (gitignored if large)
│   ├── cc_news/             raw/  +  processed/stream/  +  processed/probes/
│   ├── streaming_qa/        processed/stream/  +  processed/probes/
│   ├── temporalwiki/        processed/stream/  +  processed/probes/
│   └── tic_lm/              raw/  +  processed/stream/  +  processed/probes/
│
├── models/
│   ├── inca/
│   │   ├── config.py        ← INCAConfig dataclass (all hyperparameters)
│   │   ├── layer_manager.py ← INCALayerManager: freeze / grow transformer blocks
│   │   ├── selectors.py     ← EmbeddingQuerySelector, CrossAttentionSelector, WeightedSumSelector
│   │   ├── uclbr.py         ← UCLBRSelector (load-balance + uncertainty calibration)
│   │   ├── replay.py        ← INCAReplayBuffer (study-schedule sampling)
│   │   ├── plateau.py       ← INCAPlateauDetector (multi-signal consensus)
│   │   ├── cka.py           ← CKAMonitor (representation drift / saturation)
│   │   └── lateral.py       ← Lateral connections (Phase 2)
│   └── baselines/
│       ├── ewc.py           ← Elastic Weight Consolidation
│       ├── l2p.py           ← Learning to Prompt
│       ├── lora_moe.py      ← LoRA-MoE
│       ├── pnn.py           ← Progressive Neural Networks
│       ├── llama_pro.py     ← LlamaPro
│       ├── replay_baseline.py ← Experience Replay baseline
│       └── finetune.py      ← Plain sequential / joint fine-tune
│
├── training/
│   ├── inca_trainer.py      ← INCA continual-learning training loop
│   └── baseline_trainer.py  ← Sequential + joint FLAN-T5 baselines
│
├── scripts/
│   ├── train_inca.py        ← CLI entry: INCA
│   ├── train_baseline.py    ← CLI entry: baselines
│   ├── run_ablation.py      ← Ablation suite runner
│   └── visualize_blocks.py  ← Block-chain SVG visualiser
│
├── evaluation/
│   ├── metrics.py           ← Exact-match, BWT, FWT, CKA utilities
│   ├── probes.py            ← Diagnostic probes
│   └── eval_runner.py       ← Full evaluation harness
│
├── tests/
│   ├── test_plateau.py      ← Unit tests: plateau detector
│   ├── test_replay.py       ← Unit tests: replay buffer / study schedule
│   └── test_smoke.py        ← End-to-end smoke test (dry-run)
│
├── docs/                    ← Research PDFs & reference material
│   ├── CAPSEL_INCA_Master_Reference.pdf
│   ├── CAPSEL_Implementation_Guide.pdf
│   ├── CAPSEL_PhD_Roadmap.pdf
│   └── CAPSEL_Selector_Architecture.pdf
│
├── results/                 ← Training outputs (gitignored)
│   ├── run_<id>.log
│   └── <run_dir>/
│       ├── loss_curve.csv
│       ├── run_log.jsonl
│       └── *.pt             (checkpoints)
│
└── legacy/                  ← Archived Phase 0 code
```

---

## Datasets

Only **text-completion (seq2seq)** datasets are used for training. QA and MCQ datasets are excluded.

| ID | Local data | Periods | Coverage | Used for |
|----|-----------|---------|----------|----------|
| `cc_news` | `datasets/cc_news/` | 4 × half-year | 2017–2018 | Phase 1 primary training |
| `streaming_qa` | `datasets/streaming_qa/` | 26 × monthly | 2017–2019 | Secondary training |
| `temporalwiki` | `datasets/temporalwiki/` | 2 × annual | 2022–2023 | Evaluation |
| `tic_lm` | `datasets/tic_lm/` | 9 × daily | Apr 2019 | Paper A benchmark |
| `redpajama` | HF streaming | year-based | 2018–2023 | E-ROUTE ablation |

**Processed file schema** (each JSONL line):
```json
{"text": "...", "doc_id": "cc_news_2017_H1_000042", "period": "2017_H1",
 "source": "cc_news", "char_len": 3412}
```

**Completion framing** (applied at load time via `_base.finalise()`):
```
input_text  = "complete: " + first 50% of article text
target_text = next 200 words
```

---

## Training Modes

| Mode | Command | Description |
|------|---------|-------------|
| **INCA** | `scripts/train_inca.py` | Saturation-triggered block expansion — main contribution |
| **Sequential** | `scripts/train_baseline.py --mode sequential` | One period at a time; measures catastrophic forgetting |
| **Joint** | `scripts/train_baseline.py --mode joint` | All periods merged; upper-bound oracle |
| **Ablation** | `scripts/run_ablation.py --ablation <id>` | E-ROUTE, E-SAT, E-CLS3, E-GROW, E-PRUNE, E-SCALE |

---

## Selectors (E-ROUTE Ablation)

| Key | Class | Description |
|-----|-------|-------------|
| `embedding_query` | `EmbeddingQuerySelector` | Q = frozen embeddings; K,V from blocks **(default)** |
| `uclbr` | `UCLBRSelector` | Pre-gate + load-balance + uncertainty calibration |
| `cross_attention` | `CrossAttentionSelector` | MLP gate on mean-pooled block outputs |
| `weighted_sum` | `WeightedSumSelector` | Blind input-independent scalar per block |

---

## Growth Signals (Multi-Signal Consensus)

INCA fires `BLOCK_FULL` (grow) when **all** of:
1. **RIR** (Relative Improvement Ratio) drops below `rir_negligible` — loss plateau
2. **Grad-norm EMA** decays to `grad_norm_decay_frac × peak` — gradients vanishing
3. **CKA** ≥ `cka_saturation_threshold` — representations stopped changing

`PERIOD_LEARNED` fires early when RIR ≥ `rir_threshold` (strong learning — move on).

---

## Training UI

Start with `bash launch_ui.sh` or **Cmd+Shift+B** in VS Code.

| Page | What it shows |
|------|---------------|
| **Launch** | Dataset picker, model type, selector, hyperparameters; builds + sends CLI command |
| **Monitor** | Live SSE log stream + loss / accuracy / period-average charts |
| **Model** | Live INCA block-chain SVG + growth event history |
| **Results** | Browse `results/` file tree; render CSVs as charts |
| **History** | All past runs; click to reload in Monitor |

---

## Key Hyperparameters (`configs/inca.yaml`)

```yaml
model_name:         google/flan-t5-base
dataset:            cc_news
n_per_period:       20000       # max articles loaded per period
max_periods:        4           # set null to use all available periods
epochs_per_period:  5
lr:                 3e-4
batch_size:         32
split_frac:         0.50        # completion framing: encoder gets first half
n_max_blocks:       8           # hard cap on block chain length
k_eval:             50          # evaluate saturation every k_eval opt steps
rir_threshold:      0.30        # PERIOD_LEARNED if RIR >= this
cka_saturation_threshold: 0.95  # BLOCK_FULL if CKA >= this
replay_ratio:       0.25        # fraction of each mini-batch from replay buffer
selector:           embedding_query
```

---

## Citation

```bibtex
@misc{capsel2025,
  title  = {CAPSEL: Continual Adaptive Plasticity via Saturation-Triggered Expansion and Learning},
  author = {Kumar, Nishant},
  year   = {2025},
}
```
