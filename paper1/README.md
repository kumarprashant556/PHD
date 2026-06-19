# Paper 1 — INCA: Saturation-Driven Block Expansion for Continual Domain Adaptation

**Target venue:** TMLR (rolling) | **Timeline:** 4–6 months from now  
**Central claim:** A continually-growing transformer that freezes blocks on saturation and routes queries via token-level attention over frozen embeddings matches or exceeds fixed-architecture continual learning baselines without catastrophic forgetting, and does so without any explicit replay, EWC, or prompt engineering.

---

## Where everything lives

### Core model
| Component | Path |
|---|---|
| Block chain + growth logic | `models/inca/layer_manager.py` |
| Config (INCAConfig) | `models/inca/config.py` |
| Saturation signals (plateau, RIR, GradNorm, CKA) | `models/inca/plateau.py`, `models/inca/cka.py` |
| S-QKV selector (EmbeddingQuerySelector) | `models/inca/selectors.py` |
| Replay buffer (per-block) | `models/inca/replay.py` |
| Lateral adapters | `models/inca/lateral.py` |
| UCLBR routing (Paper 2, stubbed here) | `models/inca/uclbr.py` |

### Training
| Component | Path |
|---|---|
| INCA trainer (growth loop, replay schedule) | `training/inca_trainer.py` |
| Memory tracker | `training/memory_tracker.py` |

### Data
| Component | Path |
|---|---|
| Domain-sequential pipeline (5 domains) | `data/domain_sequential.py` |
| Tokenizer wrapper | `data/tokenizer.py` |

### Baselines (B1–B7)
| Baseline | Path |
|---|---|
| B1 Sequential fine-tune | `baselines/b1_finetune.py` |
| B2 Replay | `baselines/b2_replay.py` |
| B3 EWC | `baselines/b3_ewc.py` |
| B4 L2P | `baselines/b4_l2p.py` |
| B5 LoRA-MoE | `baselines/b5_lora_moe.py` |
| B6 LLaMA-Pro | `baselines/b6_llama_pro.py` |
| B7 PNN | `baselines/b7_pnn.py` |
| B8 BlockStack (TODO) | `baselines/b8_block_stack.py` (missing) |
| Shared runtime | `baselines/_runtime/` |

### Configs
| Config | Path |
|---|---|
| Main Paper 1 config (30K/period, 5 epochs, FLAN-T5-large) | `configs/paper_b.yaml` |
| CUDA-optimised variant | `configs/paper_b_cuda.yaml` |
| B6 baseline config | `configs/baselines/b6_paper_b.yaml` |
| B6 CUDA variant | `configs/baselines/b6_paper_b_cuda.yaml` |

### Ablation configs
| Ablation | Config | Status |
|---|---|---|
| E-TIMING — when to freeze matters | `configs/ablations/e_timing.yaml` | ✅ |
| E-ROUTE — S-QKV vs S-FULL vs S-WS | `configs/ablations/e_route.yaml` | ✅ |
| E-PRIM — G-EXPERT vs G-VERT vs G-HORIZ | `configs/ablations/e_prim.yaml` | ✅ |
| E-SAT — signal ablation (RIR/grad/CKA/plateau combos) | `configs/ablations/e_sat.yaml` | ✅ |
| E-SCOPE — lateral vs full (G-LAT) | `configs/ablations/e_scope.yaml` | ✅ |
| E-CLS3 — replay schedule (uniform vs hard-focused) | `configs/ablations/e_cls3.yaml` | ✅ |
| E-PRUNE — Fisher pruning before freeze | missing | ❌ |
| E-TRANS — transition signals (SPRT) | missing | ❌ |

### Scripts
| Script | Path |
|---|---|
| Full Paper 1 sweep (INCA + all baselines) | `scripts/run_paper_b.py` |
| Single INCA training run | `scripts/train_inca.py` |
| Inference / forward pass | `scripts/infer_inca.py` |

### Evaluation
| Component | Path |
|---|---|
| Token-level F1 metric | `evaluation/metrics.py` |

### Deployment (GPU server)
| File | Purpose |
|---|---|
| `run_cuda.py` | Cross-platform launcher (INCA + B6, multi-seed) |
| `configs/paper_b_cuda.yaml` | CUDA config |
| `Dockerfile` | Container for GPU runs |
| `requirements_cuda.txt` | Pinned deps for CUDA environment |
| `_build_cuda_zip.py` | Bundles the above into `inca_cuda_run.zip` |
| `gpu-worker-pod.yaml` | Kubernetes pod spec |

---

## Documentation
All Paper 1 docs are in `docs/papers/paper1_inca/`:

| File | Contents |
|---|---|
| `PAPER_B.md` | Paper scaffold (abstract, section stubs) |
| `paper_b_ideas_journal.md` | Running ideas log + implementation gap audit |
| `paper_b_implementation_brief.md` | Implementation notes and decisions |
| `paper_main_outline.md` | Full section-by-section paper outline |

PhD-wide planning: `docs/planning/phd_publication_plan.md`, `docs/planning/TASKS.md`

---

## Known blockers before first run
1. **🔴 Fix `chance: 0.25` in `configs/paper_b.yaml`** — MCQ periods (P3 SciQ, P4 MedMCQA) use 4-way MCQ; chance should be 0.25 not 0.0 to correctly normalise RIR.
2. **Implement `baselines/b8_block_stack.py`** — needed for the strongest non-growing baseline comparison.

## Quick start
```bash
# Full Paper 1 sweep (INCA + B1–B7, 3 seeds, 5 periods)
python scripts/run_paper_b.py

# Single ablation
python scripts/train_inca.py --config configs/ablations/e_route.yaml

# CUDA deployment
python _build_cuda_zip.py   # build zip
# then transfer inca_cuda_run.zip to GPU server and run run_cuda.py
```
