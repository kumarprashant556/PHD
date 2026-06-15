# Phase 0 — Baseline Runner & Baselines

This directory contains the unified CLI runner (`run_baseline.py`) for all
seven Phase 0 continual-learning baselines (B1–B7).  The baselines span the
main families of continual-learning strategies — forgetting floor, replay,
regularisation, prompt-based, parameter-efficient, architectural expansion,
and progressive networks — and serve as the comparison target for the Paper A
(INCA) method.

---

## Quick Start

```bash
# Run a single baseline
python Phase0/scripts/run_baseline.py --method naive
python Phase0/scripts/run_baseline.py --method ewc --lambda_ewc 100
python Phase0/scripts/run_baseline.py --method lora_moe --lora_rank 8

# Run all seven in sequence
python Phase0/scripts/run_baseline.py --method all

# Override config values on the CLI
python Phase0/scripts/run_baseline.py --method all \
    --model_name EleutherAI/pythia-2.8b \
    --dataset cc_news \
    --max_periods 6 \
    --epochs_per_period 3 \
    --batch_size 8
```

Results are written to `Phase0/results/<baseline_id>/`:

| File | Contents |
|---|---|
| `metrics.json` | Per-period pre/post PPL, probe accuracy, combined score, RIR |
| `summary.json` | ACC, BWT, FWT, final combined score |
| `training.log` | Timestamped console output |
| `config.snapshot.json` | Full config + method hyperparams at run time |
| `probes_period<N>.csv` | Per-probe results for period N |

---

## Method Names & Aliases

| CLI flag | Canonical ID | Baseline |
|---|---|---|
| `naive` | `b1_finetune` | B1 — Naive fine-tuning |
| `replay` | `b2_replay` | B2 — Experience replay |
| `ewc` | `b3_ewc` | B3 — Elastic Weight Consolidation |
| `l2p` | `b4_l2p` | B4 — Learning to Prompt |
| `lora_moe` | `b5_lora_moe` | B5 — LoRA Mixture-of-Experts |
| `llama_pro` | `b6_llama_pro` | B6 — LLaMA-Pro block expansion |
| `pnn` | `b7_pnn` | B7 — Progressive Neural Network |
| `all` | — | Run B1 → B7 in sequence |

Canonical IDs (e.g. `--method b3_ewc`) are also accepted.

---

## Model Compatibility

All baselines use `AutoModelForCausalLM` and work with any HuggingFace
causal language model.  On CUDA the model is loaded in `float16` with
`device_map="auto"` automatically.

**Recommended open-access models (no HuggingFace token required):**

```bash
# Tiny / CPU-friendly
--model_name distilgpt2                          #  82 M  (default, fast smoke tests)
--model_name gpt2                                # 117 M
--model_name gpt2-xl                             # 1.5 B

# Medium — good balance of capability and speed
--model_name EleutherAI/pythia-1.4b              # 1.4 B  ← recommended starting point
--model_name EleutherAI/pythia-2.8b              # 2.8 B  ← recommended for full runs
--model_name facebook/opt-2.7b                   # 2.7 B

# Large — needs ≥24 GB VRAM
--model_name EleutherAI/pythia-6.9b              # 6.9 B
--model_name tiiuae/falcon-7b                    # 7 B  (Apache 2.0)
--model_name facebook/opt-6.7b                   # 6.7 B
```

> **Note:** Models like `meta-llama/Llama-2-7b-hf` and
> `mistralai/Mistral-7B-v0.1` require accepting a licence on HuggingFace
> and logging in with `huggingface-cli login` before use.  All models
> listed above are fully open with no access restrictions.

---

## The Seven Baselines

### B1 — Naive Fine-tuning (`naive`)

**Source:** `Phase0/baselines/b1_finetune.py`

The catastrophic-forgetting floor.  The model is fine-tuned on each period in
turn with no memory mechanism whatsoever.  Optimiser momentum carries over
across periods, making the forgetting worse.  Every other baseline must beat
this on Backward Transfer (BWT).

Key properties:
- No extra memory or parameters.
- Fastest baseline to run.
- Expected result: very high BWT magnitude (strong forgetting).

```bash
python Phase0/scripts/run_baseline.py --method naive
```

---

### B2 — Experience Replay (`replay`)

**Source:** `Phase0/baselines/b2_replay.py`  
**Reference:** Robins (1995); iCaRL-style reservoir sampling.

Maintains a fixed-size buffer of past documents (reservoir sampling — random
subset of all seen items, capped at `buffer_size`).  Each training minibatch
is split: `(1 − replay_ratio)` of slots come from the current period and
`replay_ratio` from the buffer.

Key properties:
- Replay fraction per minibatch: `--replay_ratio` (default 0.5).
- Buffer capacity: `--buffer_size` (default 2000 documents).
- Answers: *how much of any baseline's BWT improvement comes from replay alone?*

```bash
python Phase0/scripts/run_baseline.py --method replay \
    --buffer_size 4000 --replay_ratio 0.3
```

---

### B3 — Elastic Weight Consolidation (`ewc`)

**Source:** `Phase0/baselines/b3_ewc.py`  
**Reference:** Kirkpatrick et al., 2017 — *Overcoming Catastrophic Forgetting
in Neural Networks.*  
**Implementation adapted from:** [ContinualAI/avalanche EWC plugin](https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/ewc.py) (MIT).

After each period, snapshots the diagonal empirical Fisher Information matrix
(averaged over up to `fisher_max_batches` minibatches) and a copy of the
current weights `θ*`.  During subsequent periods the loss becomes:

```
L_total = L_CE  +  (λ/2) · Σ_t Σ_i  F^t_i · (θ_i − θ*^t_i)²
```

The quadratic penalty restrains parameters that were important for past
periods from drifting.  Uses **multi-head EWC** (separate Fisher per period),
matching Avalanche's `EWCPlugin` behaviour.

Key properties:
- Regularisation weight: `--lambda_ewc` (default 100; sweep 10 / 100 / 1000).
- Fisher estimation budget: `--fisher_max_batches` (default 200).
- No extra memory for past data — pure weight regularisation.

```bash
python Phase0/scripts/run_baseline.py --method ewc --lambda_ewc 100
```

---

### B4 — Learning to Prompt (`l2p`)

**Source:** `Phase0/baselines/b4_l2p.py`  
**Reference:** Wang et al., CVPR 2022 — *Learning to Prompt for Continual
Learning.*  
**Original code:** [google-research/l2p](https://github.com/google-research/l2p)
(Apache 2.0).  PyTorch re-impl: [JH-LEE-KR/l2p-pytorch](https://github.com/JH-LEE-KR/l2p-pytorch) (MIT).

The backbone is **fully frozen**.  A small **prompt pool**
`P = {(k_m, p_m)}` is trained, where `k_m ∈ ℝᵈ` is a learnable key and
`p_m ∈ ℝ^{L×d}` is a soft-token prompt of length `L`.

For every input:
1. Query `q(x)` = mean of token embeddings.
2. Cosine-match `q(x)` against all keys → retrieve Top-N prompts.
3. Prepend retrieved prompts to the token-embedding sequence.
4. Run the frozen backbone on the extended sequence.
5. Loss = CE on original tokens + key-pull term `−⟨q, k_top⟩`.

Key properties:
- Backbone never changes — zero risk of forgetting backbone weights.
- Pool size: `--pool_size` (default 10).
- Prompt length: `--prompt_len` tokens (default 5).
- Keys retrieved per input: `--top_n` (default 3).
- Key-pull weight: `--key_pull_weight` (default 0.5).
- Works with any HF CausalLM via generic embedding-layer detection.

```bash
python Phase0/scripts/run_baseline.py --method l2p \
    --pool_size 20 --prompt_len 10 --top_n 5
```

---

### B5 — LoRA Mixture-of-Experts (`lora_moe`)

**Source:** `Phase0/baselines/b5_lora_moe.py`  
**Powered by:** [HuggingFace PEFT](https://github.com/huggingface/peft)
(Apache 2.0).  
**Design inspired by:** [LoRAMoE](https://github.com/Ablustrund/LoRAMoE)
and [MoE-LoRA](https://github.com/maidacundo/MoE-LoRA).

The backbone is **fully frozen**.  One PEFT LoRA adapter is added per period
(`expert_0`, `expert_1`, …).  Only the current period's adapter is trainable;
earlier experts are frozen.  A learned gate scalar per expert determines
how much each adapter contributes at inference.

At scoring time all adapters are merged into a single weighted adapter via
PEFT's `add_weighted_adapter` (Softmax-normalised gate weights → convex
combination of LoRA deltas).

Key properties:
- No custom layer surgery — PEFT handles Conv1D (GPT-2) and Linear
  (LLaMA/Mistral) transparently.
- LoRA rank: `--lora_rank` (default 8).
- LoRA alpha: `--lora_alpha` (default 16.0).
- Extra dependency: `pip install peft` (fails with a clear message if absent).

```bash
pip install peft
python Phase0/scripts/run_baseline.py --method lora_moe \
    --lora_rank 16 --lora_alpha 32
```

---

### B6 — LLaMA-Pro Block Expansion (`llama_pro`)

**Source:** `Phase0/baselines/b6_llama_pro.py`  
**Reference:** Wu et al., ACL 2024 — *LLaMA Pro: Progressive LLaMA with
Block Expansion.*  
**Original code:** [TencentARC/LLaMA-Pro](https://github.com/TencentARC/LLaMA-Pro)
(Apache 2.0).

At each period boundary the **last transformer block is deep-copied and
appended** to the block stack.  The output projections of the new block are
zero-initialised so the expansion is function-preserving at birth (the new
block contributes nothing until trained).  Only the newly appended block is
trained; all earlier blocks are frozen.

Adapted to be **architecture-agnostic** beyond LLaMA-2:

| Architecture | Block list | Output projections zeroed |
|---|---|---|
| LLaMA-2 / Mistral / Gemma | `model.model.layers` | `self_attn.o_proj`, `mlp.down_proj` |
| GPT-2 | `model.transformer.h` | `attn.c_proj`, `mlp.c_proj` |
| OPT | `model.model.decoder.layers` | `self_attn.out_proj`, `fc2` |
| BLOOM | `model.transformer.h` | `self_attention.dense`, `mlp.dense_4h_to_h` |
| Falcon | `model.transformer.h` | `self_attention.dense`, `mlp.dense_4h_to_4h` |

Key properties:
- Parameter count grows by one block per period (linear growth, cheaper than PNN).
- No replay buffer, no regularisation — pure depth growth.
- The reference every saturation-driven growth method (Paper A) must out-earn
  on parameter efficiency.

```bash
python Phase0/scripts/run_baseline.py --method llama_pro
```

---

### B7 — Progressive Neural Network (`pnn`)

**Source:** `Phase0/baselines/b7_pnn.py`  
**Reference:** Rusu et al., 2016 — *Progressive Neural Networks.*  
**Reference implementations:** [TomVeniat/ProgressiveNeuralNetworks.pytorch](https://github.com/TomVeniat/ProgressiveNeuralNetworks.pytorch);
[ContinualAI/avalanche PNN model](https://github.com/ContinualAI/avalanche).

One **full new column** (complete causal-LM checkpoint) is spawned per period.
Earlier columns are permanently frozen.  **Lateral adapters** (zero-init
linear projections + learned gate) feed the final hidden-state sequence of
each frozen column into the current column's input embeddings before the
transformer stack.

This is the strongest past-period preservation reference: frozen columns
guarantee zero backward transfer (BWT = 0 in theory), at the cost of
linear-in-T parameter growth.  Paper A must match this BWT while being
significantly more parameter-efficient.

Key properties:
- Each column = full copy of the base checkpoint.
- Lateral adapters adapt [ProgressiveNeuralNetworks.pytorch](https://github.com/TomVeniat/ProgressiveNeuralNetworks.pytorch)
  to sequence-level hidden-state transfer (standard LM simplification).
- Scoring uses only the last column (task-free evaluation — no task-id
  needed at test time, matching the shared harness).
- Memory: grows as `T × model_size`. Use a small model for multi-period runs.

```bash
python Phase0/scripts/run_baseline.py --method pnn --model_name distilgpt2
```

---

## CLI Reference

### Shared arguments (all methods)

| Argument | Default | Description |
|---|---|---|
| `--method` | required | `naive` / `replay` / `ewc` / `l2p` / `lora_moe` / `llama_pro` / `pnn` / `all` |
| `--config` | `Phase0/configs/base.yaml` | Path to YAML config file |
| `--dataset` | `cc_news` | Dataset name (must exist under `Phase0/data/processed/`) |
| `--model_name` | `distilgpt2` | Any HF AutoModelForCausalLM checkpoint |
| `--max_periods` | 4 | Number of temporal periods to train on |
| `--max_docs_per_period` | — | Cap documents per period (useful for quick tests) |
| `--epochs_per_period` | 5 | Training epochs per period |
| `--batch_size` | 16 | Minibatch size |
| `--lr` | 3e-5 | Learning rate |
| `--max_seq_len` | 128 | Token sequence length |
| `--seed` | 42 | Global random seed |
| `--device` | auto | `cpu` / `cuda` / `cuda:0` |

### Method-specific arguments

| Argument | Default | Method | Description |
|---|---|---|---|
| `--lambda_ewc` | 100.0 | B3 | EWC regularisation weight λ |
| `--fisher_max_batches` | 200 | B3 | Max batches for Fisher estimation |
| `--buffer_size` | 2000 | B2 | Replay buffer capacity (documents) |
| `--replay_ratio` | 0.5 | B2 | Replay fraction per minibatch |
| `--pool_size` | 10 | B4 | L2P prompt pool size M |
| `--prompt_len` | 5 | B4 | L2P prompt token length L |
| `--top_n` | 3 | B4 | L2P keys retrieved per input N |
| `--lora_rank` | 8 | B5 | LoRA rank r per expert |
| `--lora_alpha` | 16.0 | B5 | LoRA alpha scaling factor |

---

## Metrics

Each run reports the following CAPSEL metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| **PPL** | exp(mean NLL) | Perplexity on eval split — lower is better |
| **Probe Acc** | fraction correct on MC probes | Knowledge probe accuracy — higher is better |
| **Combined** | `ppl_weight × ppl_score + probe_weight × probe_acc` | Composite score (0–1, higher is better) |
| **RIR** | `(post − pre) / (1 − chance − pre)` | Relative Improvement Rate over pre-period baseline |
| **ACC** | mean of last-row in accuracy matrix | Average accuracy across all periods (final model) |
| **BWT** | mean backward transfer | Negative = forgetting; closer to 0 is better |
| **FWT** | mean forward transfer | Zero-shot accuracy on future periods before training |

---

## Dependencies

```bash
# Core (required for all baselines)
pip install torch transformers datasets pyyaml

# B5 LoRA-MoE only
pip install peft

# Optional — for faster tokenisation on large models
pip install sentencepiece protobuf
```

---

## Example: Full Pipeline

```bash
# 1. Prepare data
python Phase0/data/download_cc_news.py --max_periods 6 --max_docs_per_period 2000

# 2a. Quick run (CPU / low VRAM) — distilgpt2
python Phase0/scripts/run_baseline.py --method all \
    --model_name distilgpt2 \
    --dataset cc_news \
    --max_periods 4 \
    --epochs_per_period 3 \
    --batch_size 16

# 2b. Full run — open 2.8 B model on GPU
python Phase0/scripts/run_baseline.py --method all \
    --model_name EleutherAI/pythia-2.8b \
    --dataset cc_news \
    --max_periods 6 \
    --epochs_per_period 2 \
    --batch_size 4 \
    --max_seq_len 256

# 2c. Large-scale run — 7 B open model (needs ≥24 GB VRAM)
python Phase0/scripts/run_baseline.py --method all \
    --model_name EleutherAI/pythia-6.9b \
    --dataset cc_news \
    --max_periods 6 \
    --epochs_per_period 2 \
    --batch_size 2 \
    --max_seq_len 256

# 3. Results land in Phase0/results/
ls Phase0/results/
# b1_finetune/  b2_replay/  b3_ewc/  b4_l2p/  b5_lora_moe/  b6_llama_pro/  b7_pnn/
```
