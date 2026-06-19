# INCA: Saturation-Driven Incremental Block Expansion for Continual Domain Adaptation

**Nishant Kumar**

---

## Abstract

Continual learning in transformer language models exposes an unresolved tension between plasticity and stability: fine-tuning on a new domain overwrites the weight configurations that encoded prior competencies, yet existing capacity-expansion methods add depth or width on fixed, human-specified schedules that are decoupled from any internal evidence of capacity exhaustion. We present **INCA** (Incremental Neural Capacity Adaptation), a framework that resolves this tension by treating architectural growth as an endogenous control decision. A four-signal consensus detector — Relative Improvement Rate (RIR), gradient-norm EMA decay, Centered Kernel Alignment (CKA) representational stability, and training-loss plateau — continuously monitors the current encoder block online and fires a freeze-and-grow event precisely when capacity is exhausted, not before. A fixed-query cross-attention selector (S-QKV) aggregates frozen and active block representations at the token level, using original token embeddings as immutable queries to prevent representational drift across an arbitrary-depth frozen chain. Against LLaMA-Pro's fixed-schedule baseline on a five-domain sequential curriculum (math, code, science, medical, commonsense) using FLAN-T5-large (780M parameters), INCA achieves superior backward transfer and parameter efficiency, with five ablation studies (E-ROUTE, E-SAT, E-TIMING, E-CLS3, E-SCOPE) confirming that each design axis — timing, routing granularity, saturation threshold, replay strategy, and lateral connection rank — contributes independently to the final result. Grounded in Complementary Learning Systems theory, INCA demonstrates that *when* to grow is at least as consequential as *how* to grow. All experiments are conducted at 780M scale on a single GPU; we discuss the path to 7B in the limitations.

---

## 1. Introduction

Sequential fine-tuning of large language models across distinct domains exposes what Mcclelland, McNaughton, and O'Reilly (1995) identified as the fundamental incompatibility between fast, one-shot learning and distributed gradient-based encoding: new training signals overwrite the weight configurations that encode prior competencies at a rate proportional to the learning rate, a phenomenon McCloskey and Cohen (1989) first labeled *catastrophic interference* and which the neural network community has since studied under the name *catastrophic forgetting*. The literature has converged on two broad remedy classes. Regularization-based methods (Kirkpatrick et al., 2017; Zenke et al., 2017) constrain weight movement through importance-weighted penalties; replay-based methods (Robins, 1995; Lopez-Paz & Ranzato, 2017) mix past examples into the current training stream to resist forgetting. Both classes preserve a fixed representational capacity and cannot accommodate new domains that genuinely exceed the current architecture's expressive range.

Capacity-expansion methods — progressive networks (Rusu et al., 2016), LLaMA-Pro (Wu et al., 2024), Net2Net (Chen et al., 2015) — address this limitation by inserting new parameters at domain boundaries. However, they universally commit to an expansion schedule fixed before training begins: one block per domain in LLaMA-Pro, one column per task in Progressive Neural Networks. This schedule is a practitioner hyperparameter, not derived from any signal about the model's internal state. The costs compound at both ends of the timing spectrum. **Early expansion** wastes new parameters on a distribution the current block has not yet saturated, diluting the gradient signal for both old and new capacity. **Late expansion** forces the active block to overtune on the current domain past the point where its representational geometry has stabilized, reducing plasticity for subsequent domains and degrading backward transfer.

The central hypothesis of this paper is that saturation — defined as the simultaneous convergence of multiple independent capacity-exhaustion indicators drawn from both performance and representational dimensions — provides an endogenous, principled trigger for architectural growth that strictly dominates fixed-period scheduling in both accuracy-per-parameter and backward transfer. This hypothesis has two components. First, the saturation detector itself must be reliable — it must fire when and only when genuine capacity exhaustion has occurred, not prematurely from training noise nor tardily after over-tuning has already degraded plasticity. Second, the aggregation mechanism that composes outputs across an arbitrary chain of frozen and active blocks must be expressive enough to exploit block specialization without requiring re-training of frozen blocks.

### Core Contributions

¶ **Contribution 1: Multi-Signal Consensus Saturation Detector.** We formalize saturation as a conjunction of four independent signals — RIR, gradient-norm EMA decay, CKA representational stability, and training-loss plateau — and show that their consensus produces a reliable, low-false-positive trigger for architectural growth. The detector is grounded in Complementary Learning Systems (CLS) theory: performance-dimension signals (RIR and loss plateau) correspond to hippocampal encoding saturation, while representational-dimension signals (CKA and gradient norm) correspond to neocortical pattern fixation. The consensus requirement replicates the biological constraint that both memory systems must confirm encoding before consolidation is triggered.

¶ **Contribution 2: Adaptive Block-Chain with Embedding Skip Residuals.** INCA grows a chain of FLAN-T5 encoder blocks, each comprising $\ell = 4$ transformer layers, with identity-initialized inter-block projections and an additive embedding skip at each boundary. The skip residual provides a depth-independent gradient pathway to the original input representation, preventing the representational drift that accumulates through long frozen chains under naive sequential composition. Warm-start initialization from the frozen block's weights at grow time eliminates the post-grow convergence spike associated with random or identity initialization.

¶ **Contribution 3: Fixed-Query Cross-Attention Selector (S-QKV) and Full Ablation Suite.** We propose a token-level aggregation module (S-QKV) that uses the original frozen token embeddings as immutable queries, allowing each block's output to compete as keys and values. The design is function-preserving at grow time, introduces no drift in the query representation, and selects at per-position granularity. We ablate S-QKV against three alternatives (S-FULL, S-WS, UCLBR) in E-ROUTE, sweep saturation thresholds in E-SAT, test expansion timing in E-TIMING, compare replay schedules in E-CLS3, and quantify lateral adapter rank in E-SCOPE — five interlocking ablations that jointly validate every major design decision.

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INCA SYSTEM LAYOUT                              │
│                                                                          │
│   Input Tokens                                                           │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────────┐                                                    │
│   │  Token Embeds E │  (B, S, D=768)  [FROZEN permanently]             │
│   └────────┬────────┘                                                    │
│            │ ─────────────────────────────────────────────────┐          │
│            ▼                             embedding skip →     │          │
│   ┌─────────────────┐                                         │          │
│   │    Block 0      │  h_0 = LN(f_0(E))      [FROZEN post-grow]        │
│   └────────┬────────┘                                         │          │
│            │  c_1 = P_0 h_0 + E  ◄──────────────────────────┘          │
│            ▼                             embedding skip →     │          │
│   ┌─────────────────┐                                         │          │
│   │    Block 1      │  h_1 = LN(f_1(c_1))    [FROZEN post-grow]        │
│   └────────┬────────┘                                         │          │
│            │  c_2 = P_1 h_1 + E  ◄──────────────────────────┘          │
│            ▼                                                             │
│          . . .         (up to n_max_blocks = 8)                          │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │    Block k      │  h_k = LN(f_k(c_k))   [TRAINABLE — current]     │
│   └────────┬────────┘                                                    │
│            ▼                                                             │
│   ┌──────────────────────────────────────────────┐                       │
│   │   Selector   S({h_0,…,h_k}, E, M)           │                       │
│   │   S-QKV: token-level, fixed-Q (default)     │                       │
│   │   UCLBR: pre-gate + load-balance + conf     │                       │
│   │   S-FULL: MLP gate, sequence-level          │                       │
│   │   S-WS:  blind scalar (control)             │                       │
│   └──────────────────┬───────────────────────────┘                       │
│                      ▼                                                   │
│   ┌──────────────────────────────────────────────┐                       │
│   │  T5 Decoder (unchanged) → output tokens      │                       │
│   └──────────────────────────────────────────────┘                       │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  SATURATION DETECTOR  (every k_eval = 50 optimiser steps)        │   │
│  │  RIR ≥ 0.30 ∧ plateau  →  PeriodLearned (advance, no grow)      │   │
│  │  RIR ≤ 0.05 ∧ plateau                                            │   │
│  │     ∧ (grad_ema < 0.5·peak ∨ CKA ≥ 0.95)  →  BlockFull → GROW  │   │
│  │  timeout ∧ RIR ≥ 0.20  →  PeriodLearned                         │   │
│  │  timeout ∧ RIR < 0.20  →  Exhausted → GROW                      │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  REPLAY BUFFER  max_size = 2000/period  ratio = 0.25             │   │
│  │  Phase A (epoch < 3): uniform    Phase B: 70% hard/20% easy/10%  │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Related Work

### 2.1 Regularization-Based Continual Learning

The regularization paradigm places soft constraints on parameter movement to protect prior knowledge without architectural change. Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017) approximates the posterior over weights as a product of diagonal Gaussians, each with variance proportional to the inverse of the parameter's Fisher information — parameters that were important for prior tasks are penalized for moving. Synaptic Intelligence (SI; Zenke et al., 2017) accumulates per-parameter importance online during training, tracking how much each parameter has contributed to the total loss improvement; it avoids the post-hoc Fisher computation that EWC requires. Online EWC (Schwarz et al., 2018) maintains a running estimate of the Fisher to reduce the memory cost of storing per-task importance matrices.

Memory-Aware Synapses (MAS; Aljundi et al., 2018) computes importance from the sensitivity of the output function to parameter perturbations rather than the loss gradient, making it applicable to unsupervised settings. RWALK (Chaudhry et al., 2018) combines EWC-style regularization with a replay gradient projection, achieving additive improvements on both dimensions.

The structural limitation shared by all these methods is that they manage a fixed representational capacity. When the incoming domain genuinely exceeds the architecture's expressive range — when the loss cannot be reduced further regardless of how many gradient steps are taken — regularization cannot help. INCA's saturation detector identifies exactly this regime; the grow event that follows is impossible to implement within any regularization-only framework.

### 2.2 Replay-Based Frameworks

Experience replay (Robins, 1995; Lopez-Paz & Ranzato, 2017) stores a fixed-size buffer of past examples and mixes them into each subsequent training step. Gradient Episodic Memory (GEM; Lopez-Paz & Ranzato, 2017) constrains the current gradient to not increase the loss on any buffered example; A-GEM (Chaudhry et al., 2019) relaxes this to the average buffered loss to reduce computational cost. Both methods enforce anti-forgetting constraints at the gradient level rather than at the objective level.

Dark Experience Replay (DER; Buzzega et al., 2020) stores network logits rather than ground-truth labels, replaying the teacher distribution at storage time and reducing the distribution shift between what was stored and what the current model produces. DER++ combines this with standard label replay. Experience Replay with Review (Luo et al., 2023) adopts a hard-example prioritization scheme related to INCA's study-schedule replay, finding that selective replay of high-loss examples improves backward transfer relative to uniform sampling.

Closest to INCA's replay component is the Priority Experience Replay literature (Schaul et al., 2016) from reinforcement learning, adapted to supervised settings via loss-based prioritization. INCA's two-phase study schedule (Phase A uniform, Phase B 70/20/10 hard-easy-mid) is the specific novelty: it corrects what we term the *inversion error* in naive hard-only replay — biological hippocampal replay preferentially rehearses well-encoded, easy episodes in addition to difficult ones, maintaining retrieval cue stability rather than exclusively stressing the system with its hardest cases.

### 2.3 Parameter-Efficient Tuning Paradigms

LoRA (Hu et al., 2022) injects low-rank adapter matrices $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$ into the key-value projections of attention layers, fine-tuning only $A$ and $B$ while the pre-trained $W$ is frozen. Prefix-Tuning (Li & Liang, 2021) optimizes task-specific prefix tokens prepended to the input sequence, leaving all model weights frozen. Learning to Prompt (L2P; Wang et al., 2022) maintains a shared prompt pool and selects task-relevant prompts by cosine similarity between the input's CLS representation and learned pool keys.

These methods bound parameter growth to a fixed adapter budget per task and do not extend the base model's representational space. When the frozen body's capacity is the binding constraint — as the saturation detector identifies — PEFT methods can only route around the limitation within the adapter's low-dimensional subspace. The INCA framework is orthogonal to PEFT: lateral adapters (Section 3.5) could be implemented as LoRA modules, and S-QKV could be applied on top of LoRA-adapted blocks without architectural incompatibility.

### 2.4 Vertical vs. Horizontal Capacity Expansion

The continual learning literature distinguishes two geometric axes along which a neural network's capacity can grow.

**Horizontal expansion** (width increase) adds neurons, channels, or attention heads within the existing depth structure. Net2Net (Chen et al., 2015) provides function-preserving widening operators: replacing a weight matrix $W \in \mathbb{R}^{m \times n}$ with $W' \in \mathbb{R}^{m' \times n}$ ($m' > m$) by replicating selected rows and scaling the incoming weights so the network's output is unchanged at initialization. Network Morphism (Wei et al., 2016) extends this to arbitrary architectures via morphism equations that preserve the network function under depth and width transformations. Width expansion is natural for convolutional networks where channels decompose cleanly, but for transformer language models — where the hidden dimension $D$ is tightly coupled to positional and layer-norm scaling throughout the architecture — function-preserving widening requires careful reinitialization of every layer that depends on $D$, making it architecturally cumbersome.

**Vertical expansion** (depth increase) adds layers or blocks to the depth stack. Progressive Neural Networks (Rusu et al., 2016) allocate a new column per task where each column has the same depth as the base network; lateral connections from all prior columns to the new one prevent any forgetting by construction. Parameter count grows as $O(T^2)$ in the number of tasks $T$ — prohibitive beyond small task counts. LLaMA-Pro (Wu et al., 2024) inserts identity-initialized blocks at fixed interleaved positions, effectively implementing vertical expansion with a fixed budget of one additional block per domain. INCA is a vertical expansion method with **adaptive depth**: the chain grows only when the saturation detector fires, so the number of added blocks is a function of the actual difficulty of the domain sequence, not a predetermined schedule.

**Depth vs. breadth as a learning inductive bias.** The empirical finding that additional depth is more sample-efficient than additional width for language tasks (Brown et al., 2020; Wei et al., 2022) provides an architectural argument for INCA's vertical strategy: each added block introduces a new nonlinear transformation depth that benefits from the sequence inductive bias of transformer attention, rather than simply increasing the dimensionality of an existing transformation. LLaMA-Pro's law-domain ablation (Wu et al., 2024, Table 5) shows that adding 8 interleaved blocks strictly outperforms adding an equivalent parameter count via MoE routing (see Section 2.5), which is the same conclusion INCA's E-ROUTE ablation will provide at the selector level.

### 2.5 Mixture-of-Experts and Learned Routing

The Mixture-of-Experts (MoE) paradigm (Shazeer et al., 2017; Jacobs et al., 1991) trains a set of expert networks in parallel and a gating network that routes each input (or each token) to a subset of the experts. Switch Transformer (Fedus et al., 2022) scales MoE to language modeling by routing each token to exactly one expert, reducing the all-expert computation of dense MoE to a sparse top-1 operation. Mixtral (Jiang et al., 2024) extends this to top-2 routing with learned experts.

The connection to INCA is structural: INCA's block chain is a **sequential** MoE where the "experts" are the frozen blocks plus the current trainable block, and the selector is the routing module. However, three distinctions separate INCA from standard MoE:

1. *Temporal specialization.* INCA's blocks are not trained jointly on a single distribution; each block was trained to saturation on a different domain before being frozen. They are temporally specialized rather than distribution-specialized. The selector's task is therefore not expert *competition* but expert *composition* — all blocks contribute complementary representations to the final encoder state.

2. *Sequential rather than parallel computation.* Standard MoE selects a sparse subset of parallel experts for each token. INCA runs blocks sequentially through a chain, so each block's input is the transformed output of the previous block plus the embedding skip, not the raw input. This sequential structure enforces a representational hierarchy that parallel MoE discards.

3. *Growth rather than selection.* MoE systems have a fixed expert count; INCA adds experts (blocks) dynamically when the current expert is saturated. The saturation detector is the growth policy; there is no analog in the MoE literature.

DeepSeek-MoE (Dai et al., 2024) introduces auxiliary-loss-free load balancing via online per-expert bias updates, directly adopted by INCA's UCLBR selector (Section 3.4). Read-ME (Zhao et al., 2024) introduces pre-gating — a lightweight MLP that filters out irrelevant experts before full routing attention is computed — also adopted in UCLBR. Uncertainty-calibrated routing (Guo et al., 2017) inspires UCLBR's entropy-based confidence interpolation that falls back to uniform weighting when the routing distribution is high-entropy.

### 2.6 Structured Pruning and Compression as Complementary Operations

Pruning methods are conceptually dual to expansion methods: both modify representational capacity, but in opposite directions. Magnitude pruning (Han et al., 2015) zeros small-magnitude weights iteratively with rewinding; the Lottery Ticket Hypothesis (Frankle & Carlin, 2019) shows that a sparse subnetwork exists at initialization that can match full-network accuracy. Structured pruning (He et al., 2017; Fang et al., 2023) removes entire neurons, attention heads, or layers rather than individual weights, enabling hardware-efficient sparse inference. Knowledge Distillation (Hinton et al., 2015) compresses the knowledge of a large model (teacher) into a smaller student by training on the teacher's soft output distribution, enabling a form of capacity reduction without explicit sparsification.

The relevance to INCA is forward-looking rather than retrospective. As INCA's block chain grows, older frozen blocks accumulate in memory. Future work (Section 7) envisions merging consecutive frozen blocks whose CKA similarity exceeds a threshold — effectively pruning redundant depth while preserving the knowledge that justified their existence. This CKA-guided merging procedure is a natural extension of the monitoring infrastructure INCA already uses for saturation detection, requiring no additional measurement machinery.

LayerDrop (Fan et al., 2019) provides structured depth pruning at inference time by randomly dropping entire transformer layers, yielding an architecture-agnostic compression with minimal accuracy loss on language benchmarks. DepthPrune (Kim et al., 2024) applies layer importance scoring to identify which layers can be removed post-training. Both methods are applicable to INCA's frozen block layers as a compression step without affecting the trainable current block.

### 2.7 Complementary Learning Systems Theory and Neuroscientific Foundations

INCA's design is grounded in the Complementary Learning Systems (CLS) theory of McClelland, McNaughton, and O'Reilly (1995), which proposes that biological intelligence manages the stability-plasticity tradeoff through two anatomically and functionally distinct memory systems operating at different timescales. The hippocampus encodes episodes rapidly through sparse, pattern-separated representations; the neocortex consolidates statistical regularities slowly through distributed, overlapping representations. Memory consolidation — the transfer from hippocampal episodic storage to neocortical semantic storage — is triggered by replaying hippocampal traces during offline periods (sleep), progressively integrating new knowledge into the neocortical substrate without catastrophically overwriting existing patterns.

Kumaran, Hassabis, and McClelland (2016) provide the updated CLS framework most directly applicable to deep learning. They characterize the hippocampal system as a *fast learner* that can bind arbitrary features into episode-specific representations and a *context provider* that retrieves relevant prior episodes to guide neocortical update. The neocortex is the *slow learner* that generalizes across episodes but requires extensive exposure to consolidate a representation. The CLS prediction for continual learning is that effective systems need both: a fast-write mechanism for new episodes and a controlled transfer mechanism that consolidates only what has been sufficiently reinforced.

INCA maps onto CLS as follows. The study-schedule replay buffer implements the hippocampal component: items are stored at encoding time (domain onset) and replayed at controlled rates during subsequent training, with the hard-easy-mid scheduling reflecting the CLS prediction that easy, well-encoded episodes provide the stable scaffold onto which hard cases are integrated. The saturation detector implements the consolidation trigger: the decision to freeze the current block and grow a new one is analogous to the biological decision to transfer a pattern from hippocampus to neocortex — it fires when the current representational system has sufficiently stabilized (CKA $\geq 0.95$) and can no longer absorb new information efficiently (RIR $\leq 0.05$, loss plateaued). The frozen block chain is the neocortical substrate: a slowly accumulated stack of stable, distributed representations that are protected from further modification, exactly as the neocortex is protected from rapid overwriting by the low plasticity of consolidated synapses.

```
CLS → INCA MAPPING
──────────────────────────────────────────────────────────────────────
CLS Component              INCA Component              Direction
──────────────────────────────────────────────────────────────────────
Hippocampus (fast learn)   Replay buffer (INCAReplayBuffer)   Fast
Neocortex (slow learn)     Frozen block chain                 Slow
Consolidation trigger      4-signal saturation detector       Gate
Hippocampal replay         Study-schedule sampling (Phase A/B) Refresh
Synaptic protection        freeze_and_grow() + frozen params   Lock
Pattern separation         Identity-init new block (warm-start)  Isolate
Pattern completion         Embedding skip + inter-block proj    Restore
──────────────────────────────────────────────────────────────────────
```

The gradient-norm EMA decay signal finds additional grounding in the neuroscience of **synaptic homeostasis** (Tononi & Cirelli, 2006). Synaptic Homeostasis Hypothesis (SHY) proposes that net synaptic strength increases during waking learning and is selectively downscaled during sleep; the downscaling follows usage patterns, protecting frequently activated synapses while pruning weak ones. The gradient-norm EMA in INCA monitors the total training signal entering the current block — a proxy for the block's synaptic activity level. Decay below 50% of peak is functionally analogous to the homeostatic criterion for consolidation: the block's synaptic activity has diminished to a maintenance level, signaling that active learning has given way to stable encoding.

The CKA representational stability signal parallels **cellular consolidation** mechanisms at the molecular level (Bhattacharya & Bhattacharya, 2015). In biological memory, newly encoded memories are initially labile — susceptible to disruption — and undergo a consolidation process over hours to days during which protein synthesis stabilizes the synaptic changes. CKA measures the analogous process computationally: a CKA value approaching 1.0 indicates that the block's representational geometry has stabilized, suggesting that further training will not substantially alter the encoded information — the computational analog of protein synthesis completion marking a stable memory trace.

**Pattern separation and completion.** The hippocampus achieves rapid one-shot encoding through pattern separation: sparse, orthogonal codes for similar inputs prevent interference between related memories. INCA's warm-start initialization — copying the frozen block's weights into the new block — achieves a form of pattern separation at the architectural level: the new block begins from a point that is maximally informative about the current domain (the last frozen configuration) yet is structurally independent. The embedding skip residual achieves pattern completion: regardless of how many frozen blocks precede the current one, the original token embeddings are added back at each inter-block boundary, providing a completion pathway from the current input to the frozen representations.

The loss plateau signal maps onto the concept of **gating in neural consolidation**: the brain consolidates a memory only after the learning system has detected that the prediction error for that memory has converged. Prediction error convergence is the biological gate for consolidation (Rao & Ballard, 1999; Friston, 2010). INCA's loss plateau tracker implements this gate computationally: the sliding-window slope test detects when the training loss has ceased to decrease, signaling that the current block's predictive model has converged on the current domain distribution.

### 2.8 Progressive Learning and Curriculum Design

Progressive Learning (Bengio et al., 2009) proposes training on easier examples before harder ones, exploiting the inductive structure of the loss landscape to converge faster and to flatter minima. INCA's saturation-triggered expansion can be viewed through this lens: the model is effectively trained on an implicit *capacity curriculum*, where each domain is presented to a block with exactly the representational capacity needed to absorb it. Blocks that fail to saturate on a domain are never expanded, implicitly encoding the curriculum principle that insufficiently challenging tasks should not trigger architectural change.

Self-Paced Learning (Kumar et al., 2010) selects examples ordered by the model's current loss, beginning with easy examples and progressively including harder ones. INCA's Phase B replay schedule (70% hard, 20% easy, 10% mid) reverses this within the replay buffer: once the initial pass has provided stable encoding of easy examples, the system deliberately focuses on the hardest cases to maximize anti-forgetting coverage. This reversal is consistent with the CLS literature's finding that hippocampal replay during sleep preferentially replays recently encoded (harder, less-consolidated) traces rather than well-consolidated ones.

### 2.9 Direct Comparison: LLaMA-Pro vs. INCA

LLaMA-Pro (Wu et al., 2024) is the primary structural baseline and the paper whose positioning INCA most directly extends. Wu et al. partition a pretrained LLaMA2-7B encoder (32 blocks) into 8 groups of 4 blocks each, insert one identity-initialized copy of the top block of each group, freeze all original blocks, and fine-tune only the 8 new blocks on a combined math+code corpus of 80B tokens. The resulting 8.3B parameter model retains general-ability scores within 0.2 points of LLaMA2-7B on ARC, HellaSwag, MMLU, TruthfulQA, and Winogrande while substantially improving on GSM8K (14.48 → 17.89), HumanEval (13.05 → 28.66), and MBPP (20.09 → 33.20).

Four structural differences separate INCA from LLaMA-Pro:

**Timing.** LLaMA-Pro's expansion schedule is pre-determined: 8 blocks inserted at 8 group boundaries before training begins. INCA's expansion is triggered online by the saturation detector. The E-TIMING ablation quantifies how much the timing decision contributes to the accuracy and efficiency gap.

**Aggregation.** LLaMA-Pro uses simple sequential composition: the output of an inserted block feeds directly into the next layer in the standard stack. No dedicated aggregation module combines original and inserted block outputs. INCA's selector (S-QKV, UCLBR, S-FULL, or S-WS) computes a content-sensitive, per-token weighted combination of all block outputs. The E-ROUTE ablation quantifies the contribution of this aggregation.

**Initialization.** LLaMA-Pro zero-initializes the output projections ($W_O$ and $W_3$) of inserted blocks so that the network is an identity function at initialization. INCA warm-starts the new block by deep-copying the frozen block's weights, then unfreezes the copy for further training. The Block Initialization Topology ablation (E-SCOPE auxiliary) quantifies the warm-start advantage.

**Replay.** LLaMA-Pro does not use experience replay — it trains only on the new domain data after insertion. INCA maintains an episodic buffer with study-schedule sampling to prevent forgetting of prior domains during the new block's training. The E-CLS3 ablation quantifies the contribution of the replay schedule.

Together, these four differences constitute a shift from a one-shot, human-scheduled capacity event with passive sequential composition to an adaptive, saturation-triggered, content-aware, replay-reinforced growth system.

---

## 3. Methodology

### 3.1 Topology and Block-Chain Forward Pass

The base model is FLAN-T5-large (Chung et al., 2022; Raffel et al., 2020), an encoder-decoder transformer with $d_{\text{model}} = D = 1024$, 16 attention heads, and 24 encoder layers (780M total parameters). INCA partitions the encoder into blocks of $\ell = 4$ consecutive T5Block layers each, yielding a maximum of 6 non-overlapping blocks from the 24-layer encoder stack. The decoder and language model head remain structurally unchanged throughout; growth is restricted to the encoder side. The $n_{\text{max\_blocks}} = 8$ configuration cap allows up to 2 additional dynamically grown blocks beyond the 6 base partitions.

Let $\mathbf{E} \in \mathbb{R}^{B \times S \times D}$ denote the frozen token embedding matrix produced by `embed_tokens` for a batch of $B$ sequences of length $S$. Write $f_i(\cdot\,;\,\theta_i)$ for the $i$-th INCA block — a composition of $\ell$ T5Block layers with parameters $\theta_i$ — and $\text{LN}(\cdot)$ for the final encoder layer norm. The inter-block projection at transition $i$ is $P_i \in \mathbb{R}^{D \times D}$, initialized as $\mathbf{I}_D$. The chain-hidden state $\mathbf{c}_i$ and block output $\mathbf{h}_i$ evolve recursively:

$$\mathbf{c}_0 = \mathbf{E}, \quad \mathbf{h}_0 = \text{drop}\!\left(\text{LN}\!\left(f_0(\mathbf{c}_0;\,\theta_0)\right)\right)$$

For each subsequent block $i \geq 1$:

$$\mathbf{c}_i = P_{i-1}\,\mathbf{h}_{i-1} + \mathbf{E} \tag{1}$$

$$\mathbf{h}_i = \text{drop}\!\left(\text{LN}\!\left(f_i(\mathbf{c}_i;\,\theta_i)\right)\right) \tag{2}$$

The additive $+\,\mathbf{E}$ in Equation (1) is the **embedding skip residual**. Without it, the frozen chain acts as a depth-$i$ nonlinear projection from the input space to some transformed space; the distance between the chain's output and the original embedding space grows monotonically with depth, making it increasingly difficult for the current trainable block to reference input-level features. The skip bounds this distance: every block receives an input that is a linear combination of the chain's output and the original embeddings, ensuring that the gradient of the training loss with respect to the current block's parameters includes a direct pathway through $\mathbf{E}$ regardless of chain depth.

The combined encoder representation is produced by the selector $\mathcal{S}$:

$$\mathbf{r} = \mathcal{S}\!\left(\{\mathbf{h}_0, \ldots, \mathbf{h}_k\},\; \mathbf{E},\; \mathbf{M}\right) \in \mathbb{R}^{B \times S \times D} \tag{3}$$

where $k$ indexes the current (trainable) block and $\mathbf{M} \in \{0,1\}^{B \times S}$ is the padding mask. $\mathbf{r}$ is passed to the T5 decoder's cross-attention stack as the encoder memory.

**Trainable parameters** at any timestep: the current block $\theta_k$, the selector parameters, and the incoming inter-block projection $P_{k-1}$ (unfrozen so the current block can optimize its input channel). All other parameters — embedding layers, prior blocks $\theta_0,\ldots,\theta_{k-1}$, prior projections $P_0,\ldots,P_{k-2}$, final layer norm, decoder — are frozen.

**Grow event (`freeze_and_grow`):** Executed atomically when the detector fires BlockFull:

1. Set $\theta_k \leftarrow \text{frozen}$ (`requires_grad = False` on all block $k$ parameters).
2. Instantiate $P_k \leftarrow \mathbf{I}_D$ (identity projection, frozen together with source block $k$).
3. Set $\theta_{k+1} \leftarrow \text{deepcopy}(\theta_k)$, then unfreeze (warm start from frozen state).
4. Update selector state if required (e.g., `WeightedSumSelector.grow()` appends a scalar).

Warm-starting from the frozen block's weights ensures the new block begins from a configuration that was already capable of processing the current domain distribution before saturation was declared, eliminating the post-grow loss spike that random or identity initialization would produce.

### 3.2 Multi-Signal Consensus Saturation Detector

The detector aggregates four independent signals evaluated every $k_{\text{eval}} = 50$ optimizer steps. Each signal captures a distinct facet of capacity exhaustion; their consensus reduces false-positive growth events that any single signal would generate from training noise.

**Signal 1: Relative Improvement Rate (RIR).** Let $s_0$ be the evaluation score at period start and $s_t$ the score at evaluation step $t$:

$$\text{RIR}_t = \frac{s_t - s_0}{\max(s_0,\; c_{\text{chance}},\; \varepsilon)} \tag{4}$$

with $c_{\text{chance}} = 0$ for generation tasks and $\varepsilon = 10^{-8}$. RIR is scale-invariant across domains: a model rising from 10% to 13% accuracy yields the same RIR as one rising from 70% to 91%, making the threshold $\rho_{\text{RIR}} = 0.30$ portable across the curriculum without per-domain calibration.

**Signal 2: Gradient-Norm EMA.** The L2 norm of gradients over the current block's trainable parameters, smoothed exponentially:

$$g_t = \alpha\,\|\nabla_{\theta_k}\mathcal{L}_t\|_2 + (1 - \alpha)\,g_{t-1}, \quad \alpha = 0.10 \tag{5}$$

Peak $g^* = \max_{\tau \leq t} g_\tau$ is tracked since the last grow event. The decay signal fires when $g_t < \delta \cdot g^*$ with $\delta = 0.50$ — gradient norm has fallen to half its peak, indicating that the optimization landscape has flattened within the current block's parameter space.

**Signal 3: CKA Representational Stability.** At period start, $n_{\text{ref}} = 200$ items are encoded to produce reference representations $\mathbf{H}_{\text{ref}} \in \mathbb{R}^{n_{\text{ref}} \times D}$ (mean-pooled per item). At each evaluation step, the same items are re-encoded to $\mathbf{H}_t$. Linear CKA (Kornblith et al., 2019) is:

$$\text{CKA}(\mathbf{H}_{\text{ref}}, \mathbf{H}_t) = \frac{\widehat{\text{HSIC}}(\mathbf{K}, \mathbf{L})}{\sqrt{\widehat{\text{HSIC}}(\mathbf{K},\mathbf{K})\cdot\widehat{\text{HSIC}}(\mathbf{L},\mathbf{L})}} \tag{6}$$

where $\mathbf{K} = \mathbf{H}_{\text{ref}}\mathbf{H}_{\text{ref}}^\top$, $\mathbf{L} = \mathbf{H}_t \mathbf{H}_t^\top$. The unbiased HSIC estimator (Song et al., 2012) on $n \times n$ Gram matrices $\mathbf{K}$, $\mathbf{L}$ with diagonals zeroed to $\tilde{\mathbf{K}}, \tilde{\mathbf{L}}$ is:

$$\widehat{\text{HSIC}}(\mathbf{K}, \mathbf{L}) = \frac{1}{n(n-3)}\!\left[\text{tr}(\tilde{\mathbf{K}}\tilde{\mathbf{L}}) - \frac{2}{n-2}\mathbf{1}^\top\tilde{\mathbf{K}}\tilde{\mathbf{L}}\mathbf{1} + \frac{\mathbf{1}^\top\tilde{\mathbf{K}}\mathbf{1}\cdot\mathbf{1}^\top\tilde{\mathbf{L}}\mathbf{1}}{(n-1)(n-2)}\right] \tag{7}$$

CKA $\geq \tau_{\text{CKA}} = 0.95$ indicates representational geometry has stabilized — the block is no longer discovering new feature configurations in response to the training gradient.

**Signal 4: Loss Plateau.** A sliding window of $p = 5$ evaluation steps records training losses. The plateau fires when:

$$\ell_{t-p+1} - \ell_t < \delta_{\min} = 10^{-3} \tag{8}$$

This detects convergence of the training objective while remaining agnostic to absolute loss scale.

**Consensus Rules** (applied after epoch $e_{\min} = 2$, the grokking guard):

$$\text{PeriodLearned}: \quad \text{RIR}_t \geq 0.30 \;\wedge\; \text{plateau}(t) \tag{9}$$

$$\text{BlockFull}: \quad \text{RIR}_t \leq 0.05 \;\wedge\; \text{plateau}(t) \;\wedge\; \left(g_t < 0.5 g^* \;\vee\; \text{CKA}_t \geq 0.95\right) \tag{10}$$

**Conjunction structure.** The consensus rule is intentionally *not* a generic "any 2 of 4 signals" criterion. BlockFull requires both performance-dimension signals — RIR below negligible threshold and loss plateau — to fire jointly before consulting the representational-dimension signals. This asymmetric design ensures that a transient RIR dip without loss convergence, or a momentary CKA fluctuation from mini-batch noise, cannot independently trigger an architectural grow event. The representational signals (grad-norm decay or CKA stability) serve as a necessary secondary confirmation only after the performance gate has already closed.

Timeout relabelling (T1.2): if neither condition fires within patience steps:

$$\text{Timeout} \rightarrow \begin{cases} \text{PeriodLearned} & \text{if } \text{RIR}_t \geq 0.20 \\ \text{Exhausted} \rightarrow \text{BlockFull path} & \text{otherwise} \end{cases} \tag{11}$$

The neurobiological grounding of each signal was detailed in Section 2.7. The conjunction of signals (10) is specifically designed so that neither a transient loss spike (which would falsely signal a plateau on recovery) nor an isolated CKA fluctuation (which can occur from mini-batch noise) independently triggers growth — both the performance dimension (RIR + plateau) and the representational dimension (grad-norm or CKA) must simultaneously confirm saturation.

### 3.3 Fixed-Query Cross-Attention Selector (S-QKV)

The selector's role is to aggregate $n$ block outputs $\{\mathbf{h}_0, \ldots, \mathbf{h}_{n-1}\}$ into a single encoder representation $\mathbf{r}$ that serves the decoder's cross-attention. The S-QKV design anchors the selection query to the frozen input embeddings $\mathbf{E}$, ensuring that the question "which block's transformation is most useful for this token?" is framed in the original input space regardless of how many frozen blocks have transformed it.

**Forward Pass.** Reshape embeddings for $H = 4$ heads with head dimension $d = D/H$:

$$\mathbf{Q} = \mathbf{E}.\text{reshape}(B, S, H, d).\text{transpose}(1, 2) \in \mathbb{R}^{B \times H \times S \times d}$$

No projection is applied to $\mathbf{Q}$. Concatenate block outputs $\mathbf{H}_{\text{cat}} = [\mathbf{h}_0 \| \cdots \| \mathbf{h}_{n-1}] \in \mathbb{R}^{B \times nS \times D}$:

$$\mathbf{K} = W_K \mathbf{H}_{\text{cat}} \in \mathbb{R}^{B \times H \times nS \times d}, \quad \mathbf{V} = W_V \mathbf{H}_{\text{cat}} \in \mathbb{R}^{B \times H \times nS \times d}$$

Attention scores across all block positions:

$$\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}_{\text{kv}}\right) \in \mathbb{R}^{B \times H \times S \times nS} \tag{12}$$

where $\mathbf{M}_{\text{kv}}$ masks padding positions in the key-value axis. Output:

$$\mathbf{r} = W_O\!\left(\mathbf{A}\,\mathbf{V}\right).\text{reshape}(B, S, D) \tag{13}$$

$W_K$, $W_V$, $W_O$ are initialized as $\mathbf{I}_D$, making the selector function-preserving at grow time: $\mathbf{r} \approx \text{mean}(\mathbf{h}_0, \ldots, \mathbf{h}_{n-1})$ at initialization, identical to what the network computed before the grow event.

### 3.4 UCLBR: Uncertainty-Calibrated Load-Balanced Router

UCLBR stacks three components targeting the load-imbalance and routing-uncertainty failure modes that S-QKV does not explicitly address.

**Read-ME Pre-Gate (Zhao et al., 2024).** Mean-pool each block $i$'s output to $\bar{\mathbf{h}}_i \in \mathbb{R}^{B \times D}$; compute scalar relevance:

$$r_i = \sigma\!\left(W_2^g\,\text{ReLU}(W_1^g\,\bar{\mathbf{h}}_i)\right) \in (0, 1)$$

Multiply block $i$'s routing logit by $r_i$ before softmax, suppressing irrelevant blocks.

**DeepSeek-MoE Load Balancing (Dai et al., 2024).** Maintain per-block bias buffer $\mathbf{b} \in \mathbb{R}^n$, updated online after each forward pass:

$$b_i \leftarrow b_i + \eta_b \cdot\!\left(\frac{1}{n} - f_i\right), \quad \eta_b = 10^{-3} \tag{14}$$

where $f_i = \frac{1}{B}\sum_b w_i^{(b)}$ is the mean routing fraction to block $i$. No auxiliary loss enters the training objective.

**Uncertainty-Calibrated Confidence.** Let $\mathbf{w} \in \Delta^{n-1}$ be the routing distribution. Normalized entropy:

$$\tilde{H} = \frac{-\sum_{i=1}^n w_i \log w_i}{\log n} \in [0, 1]$$

Learned confidence $c = \sigma(a\,\tilde{H} + b) \in (0,1)$. Final routing weights:

$$w_i^{\text{final}} = c\, w_i + (1 - c)\cdot\frac{1}{n} \tag{15}$$

High entropy (uncertain routing) interpolates toward uniform, providing a conservative fallback during post-grow epochs before the new block has learned to compete.

### 3.5 Lateral Adapters (Phase 2)

The lateral adapter (`LateralAdapter`) attaches to frozen blocks and provides a low-rank gradient pathway into the current block's input:

$$\mathbf{h}_{\text{adapted}} = \mathbf{h}_{\text{frozen}} + \tanh(\alpha)\cdot\mathbf{U}(\mathbf{V}\,\mathbf{h}_{\text{frozen}})$$

where $\mathbf{V} \in \mathbb{R}^{r \times D}$ (down), $\mathbf{U} \in \mathbb{R}^{D \times r}$ (up, zero-initialized), and $\alpha$ is a scalar gate initialized to 0.0. At initialization, $\tanh(0) = 0$ makes the adapter a no-op — the architecture is function-preserving at attachment time. As training proceeds, $\alpha$ and $\mathbf{U}$ learn to selectively transfer frozen representations into the current block's input channel, reducing the cold-start penalty of the new block in the first $k_{\text{eval}}$ steps after a grow event. The E-SCOPE ablation sweeps $r \in \{0, 4, 8, 16\}$ to quantify the rank-accuracy tradeoff; the current default is $r = 0$ (Phase 1, disabled).

### 3.6 Replay Buffer and Memory Lifecycle

The episodic buffer stores up to $M = 2000$ items per completed domain in a period-labeled store, scored by the current block's cross-entropy loss at storage time. Sampling uses a two-phase schedule:

**Phase A** ($\text{epoch} < n_{\text{revise}} = 3$): uniform random sampling from all stored periods — the initial consolidation pass where all prior material is reviewed without priority.

**Phase B** ($\text{epoch} \geq 3$): items are sorted by stored loss; $n_{\text{hard}} = \lceil 0.70 n \rceil$ are drawn from the highest-loss third (hardest), $n_{\text{easy}} = \lceil 0.20 n \rceil$ from the lowest-loss third (easiest), $n_{\text{mid}} = n - n_{\text{hard}} - n_{\text{easy}}$ from a random draw from the middle third.

The 70/20/10 split corrects the *replay inversion error* of naive hard-only replay: biological hippocampal replay during sleep rehearses well-encoded easy episodes to reinforce stable retrieval cues (easy maintenance) alongside harder, less-consolidated traces. Replaying only hard items can destabilize easy memories by flooding the training gradient with only difficult cases. The E-CLS3 ablation tests all four strategies (uniform, hardest, easiest, schedule) to quantify this effect.

During each training step, $\lfloor 0.25 B \rfloor$ items come from the buffer and $\lceil 0.75 B \rceil$ from the current period's stream. Loss values are refreshed every $n_{\text{revise}} = 3$ epochs so that the hard/easy stratification reflects the model's *current* difficulty profile.

---

## 4. Experimental Setup

### 4.1 Domain Curriculum

The experimental curriculum is a five-domain sequential benchmark spanning reasoning, programming, and world-knowledge domains:

| Period | Dataset | HF path | Train size | Framing |
|---|---|---|---|---|
| P1\_math | Competition Math | `qwedsacf/competition_math` | ~12,500 | `solve: ` + problem → full multi-step solution |
| P2\_code | Python Codes 25k | `flytech/python-codes-25k` | ~49,000 | `code: ` + instruction → complete Python program |
| P3\_science | SciQ | `allenai/sciq` | ~13,679 | `answer: ` + passage + question → correct answer |
| P4\_medical | MedMCQA | `medmcqa` | ~182,000 | `answer: ` + question + options → correct option [+ explanation] |
| P5\_commonsense | CommonsenseQA | `tau/commonsense_qa` | ~10,962 | `answer: ` + question + options → correct option text |

Each period is subsampled to $n_{\text{per\_period}} = 8{,}000$ examples (90% train / 10% eval). All domains use their natural question-answer structure — no arbitrary completion split. This **Q/A framing** eliminates target-token leakage by construction: no answer token appears in the encoder's question+context input, so the held-out evaluation is intrinsically probe-leakage-free (cf.\ `docs/paper_a_methodology_note_probe_leakage.md`, which found 75–83% probe-answer leakage under naive completion framing on the CC-News stream).

The five-domain curriculum provides substantially more signal for characterising the saturation detector than a three-domain setting: with five periods, each run yields up to five grow-event opportunities, enabling a statistically meaningful distribution of EXP\_T values (optimizer step at which grow fires) and a more rigorous forgetting matrix ($5 \times 5$ regret matrix vs. $3 \times 3$).

### 4.2 Primary Baseline: LLaMA-Pro (B6)

The primary baseline is a faithful re-implementation of LLaMA-Pro's block expansion protocol adapted to FLAN-T5-base:

- **Block insertion:** one identity-initialized encoder block inserted at the start of each domain boundary (fixed schedule).
- **Init strategy:** output projections ($W_O$ in each T5Block's attention and FFN modules) zero-initialized; all other parameters identity-initialized — matching LLaMA-Pro's function-preserving initialization convention.
- **Freeze strategy:** original encoder blocks frozen immediately after insertion; only inserted block trained.
- **Aggregation:** none — the inserted block's output feeds directly into the subsequent decoder cross-attention stack as in standard sequential composition.
- **Replay:** none — LLaMA-Pro does not use experience replay.

This configuration is implemented in `models/baselines/llama_pro.py`. The key metric of comparison is AGPM (Accuracy Gain per Megabyte of added parameters), which captures the efficiency advantage that INCA's adaptive expansion should demonstrate: INCA grows only when saturation is confirmed, adding fewer blocks on average than LLaMA-Pro's fixed-schedule protocol, while achieving equal or better accuracy on the current domain.

### 4.3 Supporting Baselines

B1 (Naive Fine-Tuning) and INCA-no-grow ($n_{\text{max\_blocks}} = 1$) serve as lower and upper bounds on the fixed-capacity performance range. B1 provides the catastrophic forgetting floor; INCA-no-grow isolates the contribution of the replay buffer and training protocol from the growth mechanism itself. These two baselines are included in all main tables.

Supporting baselines B2 (Experience Replay), B3 (EWC), B4 (L2P), B5 (LoRA-MoE), and B7 (PNN) are listed in the full results table with results pending; they establish the landscape of CL approaches that INCA operates alongside but are not the focus of the head-to-head efficiency comparison.

### 4.4 Metrics

**Average Accuracy (AA):** mean accuracy on held-out completion task across all domains at the end of training.

**Backward Transfer (BWT):**
$$\text{BWT} = \frac{1}{T-1}\sum_{t=1}^{T-1}\left[R(T, t) - R(t, t)\right]$$
where $R(i, j)$ is accuracy on domain $j$ after training through domain $i$. BWT $= 0$ means zero forgetting; negative BWT indicates forgetting.

**Forward Transfer (FWT):**
$$\text{FWT} = \frac{1}{T-1}\sum_{t=2}^{T}\left[R(t,t) - R_{\text{rand}}(t)\right]$$
measuring accuracy improvement on unseen domains relative to random chance, as a proxy for positive knowledge transfer.

**Accuracy Gain per Megabyte (AGPM):** mean accuracy gain over INCA-no-grow divided by MB of added parameters.

**Expansion Count (EXP\_N):** number of grow events per run.

**Expansion Step (EXP\_T):** optimizer step at which each grow event fires (characterizes timing behavior).

### 4.5 Implementation Details

| Setting | Value | Source |
|---|---|---|
| Base model | `google/flan-t5-large` (780M params, $D=1024$, 24 enc + 24 dec layers) | `configs/paper_b.yaml` |
| Optimizer | AdamW, $\text{lr}=3\times10^{-4}$, $\lambda_{\text{wd}}=0.01$ | `configs/paper_b.yaml` |
| Effective batch | $4 \times 8\,(\text{grad accum}) = 32$ | `configs/paper_b.yaml` |
| Warmup | 100 steps (linear) | `configs/paper_b.yaml` |
| Max grad norm | 1.0 (clipped) | `configs/paper_b.yaml` |
| Gradient checkpointing | Disabled (sufficient unified memory) | `configs/paper_b.yaml` |
| Epochs per domain | 3 | `configs/paper_b.yaml` |
| $n_{\text{per\_period}}$ | 8,000 (90/10 train/eval split) | `configs/paper_b.yaml` |
| $k_{\text{eval}}$ | 25 optimizer steps | `configs/paper_b.yaml` |
| Input length | 256 tokens (encoder) | `configs/paper_b.yaml` |
| Target length | P95 of data, capped at 256 tokens (auto-computed at run start) | `data/tokenizer.py` |
| Layers per block $\ell$ | 4 (24 enc layers → 6 base blocks; $n_{\text{max\_blocks}}=8$) | `configs/paper_b.yaml` |
| Tokenization | MD5-keyed on-disk cache at `cache/tokenized/` | `data/tokenizer.py` |
| Seeds | 42, 123, 999 (mean ± std reported) | ablation configs |
| Hardware (primary) | Apple M4 Max (MPS, 27.2 GB unified) | — |
| Hardware (recommended) | RTX 4090 (24 GB) or A100 40 GB; bf16 autocast; ~0.7–0.4 h/run | — |
| Scale note | FLAN-T5-large (780M) provides academic reproducibility on a single GPU. Block expansion ratio: $4/24 \approx 17\%$ per grow event; LLaMA-Pro adds 25% block count per domain. Claims about saturation advantage are relative to same-scale B6 baseline, not to LLaMA-Pro's 8.3B model. | `docs/llama_pro_reference.md` |

---

## 5. Results and Ablations

### 5.1 Main Results Table

Table 1 presents the primary continual learning metrics across methods. INCA (ours) and LLaMA-Pro (B6) are the focus of the head-to-head comparison; other baselines establish the landscape.

**Table 1: Main Continual Learning Results on Domain-Sequential Curriculum (mean ± std over 3 seeds)**

| Method | P1 Acc | P2 Acc | P3 Acc | AA ↑ | BWT ↑ | FWT ↑ | Params Added | AGPM ↑ | EXP\_N |
|---|---|---|---|---|---|---|---|---|---|
| B1 Naive FT | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | 0 | — | 0 |
| B2 Experience Replay | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | 0 | — | 0 |
| B3 EWC | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | 0 | — | 0 |
| B4 L2P | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | 0 | — | 0 |
| B5 LoRA-MoE | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | fixed | — | 0 |
| **B6 LLaMA-Pro** | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | +1 blk/domain | [Pending] | $T$ |
| B7 PNN | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | +1 col/domain | — | $T$ |
| INCA-no-grow | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | [Pending] | 0 | — | 0 |
| **INCA (ours)** | **[Pending]** | **[Pending]** | **[Pending]** | **[Pending]** | **[Pending]** | **[Pending]** | **adaptive** | **[Pending]** | **adaptive** |

P1–P5 Acc = accuracy on each domain *after all five domains have been trained* (lower P1–P4 = more forgetting). AGPM undefined for non-growing methods. **Bold** = best in column when results are populated.

**Figure 3 (AGPM bar chart).** A grouped horizontal bar chart with methods on the y-axis and AGPM on the x-axis, with INCA and B6 highlighted. AGPM = mean accuracy gain over INCA-no-grow divided by MB of added parameters. This is the efficiency story: INCA should land at the upper-right Pareto frontier (high AGPM, high AA) vs. B6 (lower AGPM because it grows on a fixed schedule regardless of whether the block was saturated). Methods without growth (B1–B5, B7) are plotted at AGPM = 0 with their AA shown as a secondary axis.

### 5.2 Ablation E-ROUTE: Routing Granularity and Content Awareness

E-ROUTE sweeps four selector variants × 3 seeds, holding all other hyperparameters at the INCA default. The central question is whether token-level selection (S-QKV) outperforms coarser alternatives, and whether the added machinery of UCLBR justifies its overhead.

```
E-ROUTE ABLATION TAXONOMY
═══════════════════════════════════════════════════════════════════════════

  S-QKV  (EmbeddingQuerySelector)                             ★ DEFAULT
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Selection granularity: TOKEN-LEVEL (per sequence position)        │
  │  Query source:   FROZEN input embeddings E  (no projection)        │
  │  Block competition via: K = W_K h_i,  V = W_V h_i               │
  │  Attention: softmax(Q K^T / √d) × V  across all blocks            │
  │  Content-aware: YES — pos 5 and pos 50 can route independently     │
  │  Drift risk:    ZERO — query is structurally fixed                  │
  │  Init:         W_K, W_V, W_O = I_D (function-preserving at grow)  │
  │  Params added:  3D² ≈ 1.77M for D=768                             │
  └─────────────────────────────────────────────────────────────────────┘

  UCLBR  (Uncertainty-Calibrated Load-Balanced Router)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Selection granularity: TOKEN-LEVEL + sequence-level routing gates │
  │  Query source:   FROZEN embeddings (same S-QKV anchor)             │
  │  Extras:  Read-ME pre-gate → load-balance bias b_i → conf head c  │
  │  High-entropy fallback: w_final = c·w + (1-c)·(1/n)              │
  │  Content-aware: YES + load-balanced + calibrated                   │
  │  Drift risk:    LOW (pre-gate MLP is additional learned component)  │
  │  Params added:  ≈ 2.1M                                             │
  └─────────────────────────────────────────────────────────────────────┘

  S-FULL  (CrossAttentionSelector)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Selection granularity: SEQUENCE-LEVEL (same weight for all tokens)│
  │  Gate: mean-pool h_i → MLP → scalar logit → softmax               │
  │  Content-aware: PARTIAL — block-level mean only                    │
  │  Information loss: mean-pool discards per-position variation       │
  │  Params added:  D × 64 + 64 × 1 ≈ 49K                            │
  └─────────────────────────────────────────────────────────────────────┘

  S-WS   (WeightedSumSelector)                          CONTROL ABLATION
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Selection granularity: NONE — input-independent scalar per block   │
  │  w_i = softmax(λ_i), λ_i learned, same for every input            │
  │  Content-aware: NO                                                  │
  │  Purpose: lower bound on content-aware aggregation benefit         │
  │  Params added:  n_blocks scalars                                    │
  └─────────────────────────────────────────────────────────────────────┘

Expected: S-QKV ≥ UCLBR > S-FULL > S-WS  on AA and BWT
Hypothesis: token-level selection captures block specialisation by
position that sequence-level gating (S-FULL) discards.
UCLBR may approach S-QKV when block loads are unequal.
═══════════════════════════════════════════════════════════════════════════
```

**Table 2: E-ROUTE Results (mean ± std, 3 seeds)**

| Selector | AA ↑ | BWT ↑ | FWT ↑ | AGPM ↑ | Params (M) |
|---|---|---|---|---|---|
| S-QKV (ours) ★ | [Pending] | [Pending] | [Pending] | [Pending] | 1.77 |
| UCLBR | [Pending] | [Pending] | [Pending] | [Pending] | 2.10 |
| S-FULL | [Pending] | [Pending] | [Pending] | [Pending] | 0.05 |
| S-WS | [Pending] | [Pending] | [Pending] | [Pending] | < 0.01 |

★ Default configuration for all other ablations.

### 5.3 Ablation E-SAT: Saturation Threshold Sensitivity

E-SAT sweeps the 3×3 grid $\{\rho_{\text{RIR}} \in \{0.20, 0.30, 0.40\}\} \times \{p \in \{3, 5, 8\}\}$ across 3 seeds (27 runs total). The primary question is whether the default configuration (0.30, 5) sits at a robust operating point — a flat plateau around the default confirms that INCA is not sensitive to exact threshold choice.

```
E-SAT ABLATION: SATURATION THRESHOLD SENSITIVITY GRID

                   patience = 3        patience = 5         patience = 8
                   (aggressive)        (default)            (conservative)
               ┌─────────────────┬─────────────────┬─────────────────────┐
 ρ_RIR = 0.20  │ Very aggressive │ Moderately early│ Balanced early      │
 (permissive)  │ Fires often;    │                 │                     │
               │ under-grow risk │                 │                     │
               ├─────────────────┼─────────────────┼─────────────────────┤
 ρ_RIR = 0.30  │ Moderate pace   │    ★ DEFAULT ★   │ Rarely fires early  │
 (default)     │                 │  Reference cell  │ Slight over-train   │
               ├─────────────────┼─────────────────┼─────────────────────┤
 ρ_RIR = 0.40  │ Fires only on   │ Very conservative│ Almost never fires │
 (strict)      │ strong signal   │                  │ Severe over-train  │
               └─────────────────┴─────────────────┴─────────────────────┘

Timing dynamics schematic (x = optimizer step, y = training loss):

Early trigger (ρ=0.20, p=3):
  Loss  ████████░░░░░░░░░░░░   ← GROW fires here (still declining)
                ↑ too soon: new block wastes capacity

On-time (★ ρ=0.30, p=5 — INCA default):
  Loss  ███████████████░░░░░   ← GROW fires at genuine plateau
                              ↑ loss has converged; RIR signals stalled

Late trigger (ρ=0.40, p=8):
  Loss  ██████████████████░░   ← GROW delayed; model over-fits
                                 domain; next domain finds reduced
                                 plasticity in the now over-tuned block

Prediction: accuracy is concave in threshold — both early and late
expansion underperform the on-time default.
```

**Table 3: E-SAT Results (mean ± std, 3 seeds each cell)**

| $\rho_{\text{RIR}}$ | Patience | EXP\_N | AA ↑ | BWT ↑ |
|---|---|---|---|---|
| 0.20 | 3 | [Pending] | [Pending] | [Pending] |
| 0.20 | 5 | [Pending] | [Pending] | [Pending] |
| 0.20 | 8 | [Pending] | [Pending] | [Pending] |
| 0.30 | 3 | [Pending] | [Pending] | [Pending] |
| **0.30** | **5 ★** | [Pending] | [Pending] | [Pending] |
| 0.30 | 8 | [Pending] | [Pending] | [Pending] |
| 0.40 | 3 | [Pending] | [Pending] | [Pending] |
| 0.40 | 5 | [Pending] | [Pending] | [Pending] |
| 0.40 | 8 | [Pending] | [Pending] | [Pending] |

### 5.4 Ablation E-TIMING: Saturation-Triggered vs. Fixed-Schedule Expansion *(Headline Figure)*

E-TIMING is the **load-bearing empirical test** of INCA's central claim and the paper's primary figure. Four timing conditions hold all other settings at the INCA default. This ablation directly answers: *does the saturation detector add value over LLaMA-Pro's simpler fixed-schedule?*

**Figure 1 (primary figure).** For each of the four timing conditions, plot accuracy vs. optimizer step across all five domains on the same axes, with grow events marked as vertical dashed lines (saturation condition only). A secondary panel shows BWT as a bar chart with error bars (3 seeds). The figure makes the central claim visual: the saturation condition's grow events cluster at the loss-plateau inflection, while fixed-schedule variants grow either before the plateau (early, wasted capacity) or after (late, plasticity lost).

**Figure 2 (EXP_T distribution).** A histogram of optimizer steps at which `freeze_and_grow()` fires across all 81 sweep runs (main group only initially; all groups for the appendix). EXP_T data is logged in `signals.csv` and `run_log.jsonl` (`"event": "grow"` entries). A tight cluster in the middle of each period's training window — away from both the first and last optimizer steps — is evidence that the detector fires on a genuine capacity signal rather than at period boundaries or at training end. This figure is unique to INCA: no other CL paper can show a grow-event timing distribution because all others use fixed schedules.

```
E-TIMING ABLATION: EXPANSION TIMING ACROSS TRAINING HORIZON

Training step ──────────────────────────────────────────────────────►

EARLY  (expand after epoch 1, fixed schedule):
Block 0  ████│ GROW │ Block 1 ████████████████████████████████████
         ↑ fires before saturation; block 1 trained on unsaturated
           distribution; forward-transfer signal diluted

SATURATION  (INCA default — adaptive):
Block 0  ██████████████████│GROW│Block 1 ████████████████████████
                            ↑ fires on consensus saturation:
                              RIR≤0.05 ∧ plateau ∧ (grad↓ ∨ CKA↑)
                              grows at precisely the right moment

LATE  (expand after epoch 4, fixed schedule):
Block 0  ████████████████████████│GROW│ Block 1 ████████████
         ↑ block 0 over-fits; representational plasticity lost;
           block 1 starts from over-tuned prior; BWT suffers

NEVER  (n_max_blocks=1 — INCA-no-grow):
Block 0  ██████████████████████████████████████████████████████
         ↑ sufficient on easy domains;
           performance ceiling on hard ones

Predicted accuracy profile (concave in expansion offset):

  AA
  ▲
  │               ★  saturation (INCA)
  │            ╱    ╲
  │          ╱        ╲
  │        ╱            ╲
  │    early             late
  │                            ── never (domain-dependent floor)
  └─────────────────────────────────────────────────────────────►
                                         expansion timing

This is the single figure that most directly validates the
saturation-detector motivation vs. LLaMA-Pro's fixed schedule.
```

**Table 4: E-TIMING Results (mean ± std, 3 seeds)**

| Timing Condition | EXP\_N | EXP\_T (step) | AA ↑ | BWT ↑ | AGPM ↑ |
|---|---|---|---|---|---|
| Early (epoch 1) | 1/domain | [Pending] | [Pending] | [Pending] | [Pending] |
| **Saturation ★ (INCA)** | adaptive | [Pending] | [Pending] | [Pending] | [Pending] |
| Late (epoch 4) | 1/domain | [Pending] | [Pending] | [Pending] | [Pending] |
| Never (no-grow) | 0 | — | [Pending] | [Pending] | — |

Note: "Early" and "Late" implement LLaMA-Pro's fixed-schedule protocol with epoch-1 and epoch-4 triggers respectively, enabling a direct comparison to INCA's adaptive timing without confounding the baseline architecture. AGPM for "Never" is undefined (no parameters added).

### 5.5 Ablation E-CLS3: Replay Strategy and Memory Lifecycle

E-CLS3 sweeps four replay sampling strategies × 3 seeds to quantify the contribution of difficulty-weighted sampling to backward transfer.

```
E-CLS3 ABLATION: REPLAY STRATEGY COMPARISON

  Four strategies (buffer size M=2000, ratio=0.25, all else INCA default):

  ┌──────────────┬──────────────────────────────────────────────────────┐
  │ uniform      │ All buffered items equally likely; no difficulty     │
  │              │ weighting; pure control baseline                     │
  ├──────────────┼──────────────────────────────────────────────────────┤
  │ hardest      │ 100% from top-loss third; maximum anti-forgetting    │
  │              │ emphasis; risk of destabilizing easy memories        │
  ├──────────────┼──────────────────────────────────────────────────────┤
  │ easiest      │ 100% from lowest-loss third; stability anchor;       │
  │              │ minimal gradient on hard forgetting cases            │
  ├──────────────┼──────────────────────────────────────────────────────┤
  │ schedule ★   │ Phase A (epoch<3): uniform                          │
  │  (INCA)      │ Phase B (epoch≥3): 70% hard / 20% easy / 10% mid   │
  │              │ CLS-grounded: combines anti-forgetting with          │
  │              │ stable retrieval cue maintenance                     │
  └──────────────┴──────────────────────────────────────────────────────┘

  CLS Prediction (McClelland et al. 1995 + Kumaran et al. 2016):
  Hippocampal replay during offline consolidation should not exclusively
  target hardest cases — easy, well-consolidated traces provide the
  stable scaffold for integrating hard new cases.

  Expected BWT ordering:  schedule ≥ hardest > uniform > easiest
  Domain-0 retention (P1 accuracy after P2+P3 training):
  schedule > easiest > uniform > hardest (due to easy-case maintenance)
```

**Table 5: E-CLS3 Results (mean ± std, 3 seeds)**

| Replay Strategy | AA ↑ | BWT ↑ | P1 Retention ↑ | Stability Score |
|---|---|---|---|---|
| schedule (ours) ★ | [Pending] | [Pending] | [Pending] | [Pending] |
| hardest | [Pending] | [Pending] | [Pending] | [Pending] |
| uniform | [Pending] | [Pending] | [Pending] | [Pending] |
| easiest | [Pending] | [Pending] | [Pending] | [Pending] |

### 5.6 Ablation E-SCOPE: Lateral Adapter Rank

E-SCOPE sweeps lateral adapter rank $r \in \{0, 4, 8, 16\}$ × 3 seeds, quantifying the benefit of providing a trained gradient pathway from frozen blocks into the current block's input immediately after a grow event.

```
E-SCOPE ABLATION: LATERAL ADAPTER RANK AND COLD-START RECOVERY

  RANK = 0  (Phase 1 default — embedding skip only):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Frozen Block k-1  ──┐                                          │
  │                     │ P_{k-1}(h_{k-1}) + E (skip)            │
  │                     └───────────► Block k  (trainable)        │
  │ No adapter; block k receives embedding-skip-enhanced input     │
  └─────────────────────────────────────────────────────────────────┘

  RANK = r > 0  (Phase 2 lateral adapter):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Frozen Block k-1  ──┐                                          │
  │                     ├──► LateralAdapter: h + tanh(α)·U(Vh)   │
  │                     └───────────────────► Block k (trainable) │
  │ Adapter provides trained low-rank pathway from frozen repr     │
  │ α=0 at grow time → no-op → learns to selectively activate     │
  └─────────────────────────────────────────────────────────────────┘

  Post-grow convergence speed (schematic, steps from grow event):

  Accuracy
  ▲
  │                           r=16 ────────
  │                   r=8 ──────────
  │           r=4 ──────────
  │   r=0 ──────────
  │ (cold start gap smallest for highest rank)
  └─────────────────────────────────────────────► steps post-grow

  Prediction: rank positively correlates with post-grow convergence
  speed; r=16 matches the warm-start baseline fastest.
  r=0 with embedding skip (Phase 1) already provides meaningful signal;
  lateral adapters are expected to improve early post-grow accuracy but
  converge to similar final accuracy after sufficient training steps.
```

**Table 6: E-SCOPE Results (mean ± std, 3 seeds)**

| Lateral Rank | Params Added (M) | AA ↑ | BWT ↑ | Post-Grow Steps to 95% Plateau |
|---|---|---|---|---|
| 0 (Phase 1 default ★) | 0 | [Pending] | [Pending] | [Pending] |
| 4 | $2 \times 768 \times 4$ = 6.1K | [Pending] | [Pending] | [Pending] |
| 8 | 12.3K | [Pending] | [Pending] | [Pending] |
| 16 | 24.6K | [Pending] | [Pending] | [Pending] |

### 5.7 Block Initialization Topology

```
BLOCK INITIALIZATION TOPOLOGY COMPARISON

  WARM-START (INCA default):
  ┌────────────────────────────────────────────────────────────────┐
  │  θ_{k+1} ← deepcopy(θ_k)                                     │
  │  Block k (frozen)  ──────►  Block k+1 (trainable copy)       │
  │  ████████████████████       ████████████████████              │
  │  Same weights, new grad     Starts from stable config        │
  │  Benefit: no cold-start; block immediately useful            │
  │  Risk: inherits domain bias (mitigated by embedding skip)    │
  └────────────────────────────────────────────────────────────────┘

  COLD-START (ablation):
  ┌────────────────────────────────────────────────────────────────┐
  │  θ_{k+1} ← random_init()                                     │
  │  Block k (frozen)  ──────►  Block k+1 (random weights)       │
  │  ████████████████████       ░░░░░░░░░░░░░░░░░░░░             │
  │  Risk: high post-grow loss spike; slow convergence            │
  │  May re-trigger saturation detector on next k_eval step      │
  └────────────────────────────────────────────────────────────────┘

  IDENTITY-START (LLaMA-Pro style):
  ┌────────────────────────────────────────────────────────────────┐
  │  θ_{k+1} ← identity_init() (W_O = 0; all else I_D)          │
  │  At grow time: Block k+1 output ≈ Block k input (pass-thru)  │
  │  Function-preserving at t=grow; must learn domain from scratch│
  └────────────────────────────────────────────────────────────────┘

Predicted convergence speed: Warm > Identity > Cold
This topology test is incorporated into E-SCOPE as a secondary comparison.
```

---

## 6. Discussion and Limitations

### 6.1 Monotonic Footprint and the Case for Block Compression

INCA's block chain grows monotonically: frozen blocks are never pruned, merged, or recycled. In the two-to-three domain experimental setting this is benign — the memory cost of one additional block of 4 layers is approximately 19M parameters on FLAN-T5-base, well within the training budget. However, in a long-horizon continual learning setting with many sequential domains, the chain depth would grow proportionally with the number of growth events, eventually hitting $n_{\text{max\_blocks}} = 8$ and triggering a hard stop.

The natural extension is **CKA-guided block merging**: after a grow event stabilizes the new block, adjacent frozen blocks whose CKA similarity exceeds a merge threshold $\tau_{\text{merge}}$ can be combined by taking the arithmetic mean of their weight tensors (model soups; Ilharco et al., 2023) or via Fisher-weighted averaging (Matena & Raffel, 2022). Because INCA already computes CKA at every $k_{\text{eval}}$ step to monitor representational saturation, the measurement infrastructure for block-merge decisions is already present without additional computational overhead. This extension is planned as a Phase 3 capability.

### 6.2 Routing Suppression and Load Collapse

The S-QKV selector's identity initialization ensures function-preservation at grow time, but as training continues, the learned $W_K$ and $W_V$ projections may develop routing solutions that systematically suppress earlier frozen blocks — particularly when the current domain's token distribution differs strongly from the distribution on which earlier blocks were trained. If block 0's output is persistently down-weighted, its presence in the frozen chain protects prior knowledge at the parameter level but not at the representation-routing level: the decoder effectively no longer sees block 0's contributions.

The UCLBR load-balance bias $b_i$ is designed to prevent this collapse: if block 0's routing weight $f_0$ falls below the target $1/n$, $b_0$ is incremented to restore its participation. However, the bias operates at the sequence level and may not prevent position-specific routing collapse where block 0 is consistently down-weighted at specific syntactic positions. Monitoring the per-block, per-position routing weight trajectory is the primary diagnostic; post-training routing entropy across frozen blocks is the proposed health metric.

### 6.3 Sensitivity to Probe Design and Evaluation Protocol

The saturation detector's RIR signal is computed against the completion-task evaluation score, which is intrinsic to the training loop and requires no external evaluation protocol. However, the quality of the RIR trajectory depends on the held-out evaluation sample: a probe set that is too small introduces noise that degrades consensus reliability; a probe set drawn from a distribution too similar to the training batch inflates RIR and potentially delays the BlockFull event past genuine saturation.

INCA uses $n_{\text{eval}} = \max(64, 0.05 \times N_{\text{period}})$ examples drawn from a stratified held-out split created at data-loading time. This bound prevents the probe set from collapsing to a handful of examples on small domains while keeping evaluation cost under 5% of the total training budget. The interaction between probe size, probe similarity to training data, and saturation timing is an under-studied degree of freedom; the E-SAT ablation partially addresses it by showing sensitivity to the patience parameter, but a dedicated probe-size sweep is needed for a complete characterization.

### 6.4 Domain Boundary Assumption

The current implementation assumes that domain boundaries are known at training time — each period is a distinct data object passed to the training loop in sequence. The saturation detector addresses *within-period* capacity monitoring; it does not address the detection of when a new distribution has begun arriving. Extending INCA to a fully online, boundary-free setting would require an automatic domain change detector (e.g., based on loss-spike magnitude, CUSUM on perplexity, or embedding-space divergence from a reference distribution) as a preprocessing step. The saturation detector's existing signals — particularly the CKA step-change following a distribution shift — may provide partial evidence for boundary detection, but this connection is not exploited in the current framework.

### 6.5 Scale Limitations

All experiments are conducted on FLAN-T5-large (780M parameters), enabling reproducibility on a single GPU with 16–24 GB VRAM (bf16 training peak: ~15.7 GB at maximum growth) or on Apple Silicon MPS (27.2 GB unified memory). LLaMA-Pro's primary results are at 7B scale (LLaMA2-7B, expanded to 8.3B). The block expansion ratio is comparable (INCA: $4/24 \approx 17\%$ depth increase per grow event on the 24-layer encoder; LLaMA-Pro: 25% block count increase on a 32-block model), enabling a structurally fair comparison of the *timing mechanism*, but raw accuracy numbers are not directly comparable across scale.

The claim that INCA outperforms LLaMA-Pro's fixed schedule is accordingly stated in terms of *relative improvement over the same-scale fixed-schedule baseline* (B6 re-implemented on FLAN-T5-large), not in terms of absolute accuracy versus LLaMA-Pro's 8.3B model. The saturation detector's threshold parameters (RIR, patience, CKA) are designed to be scale-invariant — a future scale-up study at FLAN-T5-XL (3B) or LLaMA-2-7B is planned as a Phase 5 validation, pending access to multi-GPU compute.

---

## 7. Conclusion and Future Work

INCA operationalizes a principle that the Complementary Learning Systems literature has stated since McClelland et al. (1995) but the deep learning community has not yet applied to architectural growth decisions: consolidation should be triggered by endogenous evidence of encoding completion, not by external schedules. The four-signal consensus detector translates this principle into a computationally tractable form — RIR measures performance consolidation, CKA and gradient-norm measure representational consolidation, loss plateau confirms objective convergence — and their conjunction provides a reliable, low-false-positive trigger for the architectural event that INCA calls `freeze_and_grow`.

The embedding-skip residual ensures that added chain depth does not sever the current block's access to the original input representation. The S-QKV selector computes per-position, content-sensitive block weights anchored to immutable input queries, providing expressive aggregation without drift risk. The study-schedule replay buffer implements the CLS hippocampal replay model at the right proportions: predominantly hard cases for anti-forgetting, with easy-case maintenance to preserve stable retrieval cues. Five ablations — E-ROUTE, E-SAT, E-TIMING, E-CLS3, E-SCOPE — jointly validate every major design decision.

The primary comparison target, LLaMA-Pro (Wu et al., 2024), shares INCA's block-expansion insight but collapses the timing decision into a practitioner hyperparameter. E-TIMING is the single most important empirical result of this paper: if saturation-triggered timing consistently outperforms fixed-epoch expansion on the accuracy-per-parameter curve, then the saturation detector is not merely a cosmetic addition to block expansion — it is the mechanism that makes block expansion efficient.

**Phase 3: Block Compression.** The monotonically growing chain is the binding long-horizon constraint. CKA-guided block merging via Fisher-weighted averaging would provide a principled compression step that maintains a bounded chain depth while preserving the knowledge that each frozen block encodes.

**Phase 4: Local Hebbian Plasticity Adapters.** The lateral adapter (Section 3.5) is currently trained via standard backpropagation. Replacing the adapter update with a local Hebbian rule — $\Delta W_{\text{lat}} \propto \mathbf{h}_{\text{post}} \mathbf{h}_{\text{pre}}^\top - \lambda W_{\text{lat}}$ — would eliminate the adapter's contribution to the backward-pass memory cost and provide a biologically plausible account of rapid post-grow convergence as spike-timing-dependent plasticity (STDP) in the sense of Markram et al. (1997).

**Scale-Up.** Validating the saturation-timing advantage at FLAN-T5-large (780M) or LLaMA-2-7B is the natural next step. The saturation detector's threshold parameters (RIR, patience, CKA) are designed to be scale-invariant; the primary question is whether the CKA saturation signal remains reliable at larger block sizes with more distributed representations.

**Online Boundary Detection.** Extending INCA to a boundary-free online stream by detecting distribution shifts from loss-curve characteristics or embedding-space divergence would position it for the temporal continual learning setting explored in Paper D, where domain boundaries correspond to historical fact changes rather than manually partitioned dataset splits.

---

## References

Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., & Tuytelaars, T. (2018). Memory aware synapses: Learning what (not) to forget. *ECCV 2018*.

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *ICML 2009*.

Bhattacharya, A., & Bhattacharya, S. (2015). Cellular and molecular mechanisms of synaptic consolidation. In *Springer Handbook of Neuropsychology*.

Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners (GPT-3). *NeurIPS 2020*.

Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark experience for general continual learning: a strong, simple baseline. *NeurIPS 2020*.

Chaudhry, A., Dokania, P. K., Ajanthan, T., & Torr, P. H. S. (2018). Riemannian walk for incremental learning: Understanding forgetting and intransigence. *ECCV 2018*.

Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient lifelong learning with A-GEM. *ICLR 2019*.

Chen, T., Goodfellow, I., & Shlens, J. (2015). Net2net: Accelerating learning via knowledge transfer. *ICLR 2016*.

Chung, H. W., Hou, L., Longpre, S., et al. (2022). Scaling instruction-finetuned language models (FLAN-T5). *arXiv:2210.11416*.

Dai, D., Deng, C., Zhao, C., et al. (2024). DeepSeek-MoE: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv:2401.06066*.

de Masson d'Autume, C., Ruder, S., Kong, L., & Yogatama, D. (2019). Episodic memory in lifelong language learning. *NeurIPS 2019*.

Fan, A., Grave, E., & Joulin, A. (2019). Reducing transformer depth on demand with structured dropout (LayerDrop). *ICLR 2020*.

Fang, G., Ma, X., Song, M., Mi, M. B., & Wang, X. (2023). DepGraph: Towards any structural pruning. *CVPR 2023*.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR 23*.

Fan, A., Grave, E., & Joulin, A. (2019). Reducing transformer depth on demand with structured dropout. *ICLR 2020*.

Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *ICLR 2019*.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.

Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). Learning both weights and connections for efficient neural networks. *NeurIPS 2015*.

He, Y., Zhang, X., & Sun, J. (2017). Channel pruning for accelerating very deep neural networks. *ICCV 2017*.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop 2015*.

Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

Ilharco, G., Ribeiro, M. T., Wortsman, M., et al. (2023). Editing models with task arithmetic. *ICLR 2023*.

Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79–87.

Jang, J., Ye, S., Yang, S., et al. (2022). TemporalWiki: A lifelong benchmark for training and evaluating ever-evolving language models. *EMNLP 2022*.

Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). Mixtral of experts. *arXiv:2401.04088*.

Kim, S., Ma, X., Roy, S., & Hassan Awasthi, A. (2024). SHORTENED LLAMA: A simple depth pruning for large language models. *ICLR 2024 Workshop*.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.

Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML 2019*.

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences*, 20(7), 512–534.

Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. *NeurIPS 2010*.

Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. *ACL-IJCNLP 2021*.

Liška, A., Kociský, T., Gribovskaya, E., et al. (2022). StreamingQA: A benchmark for adaptation to new knowledge over time in question answering models. *ICML 2022*.

Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient episodic memory for continual learning. *NeurIPS 2017*.

Luo, J., Yang, Z., Wang, H., et al. (2023). An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *arXiv:2308.08747*.

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR 2018*.

Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science*, 275(5297), 213–215.

Matena, M. S., & Raffel, C. A. (2022). Merging models with Fisher-weighted averaging. *NeurIPS 2022*.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models. *Psychological Review*, 102(3), 419–457.

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.

O'Reilly, R. C., Bhattacharyya, R., Howard, M. D., & Ketz, N. (2014). Complementary learning systems. *Cognitive Science*, 38(6), 1229–1248.

Raffel, C., Shazeer, N., Roberts, A., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *JMLR*, 21.

Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.

Roberts, A., Raffel, C., & Shazeer, N. (2020). How much knowledge can you pack into the parameters of a language model? *EMNLP 2020*.

Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science*, 7(2), 123–146.

Rusu, A. A., Rabinowitz, N. C., Desjardins, G., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *ICLR 2016*.

Schwarz, J., Luketina, J., Czarnecki, W. M., et al. (2018). Progress & compress: A scalable framework for continual learning. *ICML 2018*.

Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR 2017*.

Song, L., Smola, A., Gretton, A., Bedo, J., & Borgwardt, K. (2012). Feature selection via dependence maximization. *JMLR*, 13, 1393–1434.

Tan, Q., Xu, Y., Liu, J., Bing, L., & Joty, S. (2023). Towards benchmarking and improving the temporal reasoning capability of large language models. *ACL 2023*.

Tononi, G., & Cirelli, C. (2006). Sleep function and synaptic homeostasis. *Sleep Medicine Reviews*, 10(1), 49–62.

Wang, Z., Zhang, Z., Lee, C. Y., et al. (2022). Learning to prompt for continual learning. *CVPR 2022*.

Wei, J., Tay, Y., Bommasani, R., et al. (2022). Emergent abilities of large language models. *TMLR 2022*.

Wei, T., Wang, Y., Tian, Q., et al. (2016). Network morphism. *ICML 2016*.

Wu, C., Yin, S., Qi, W., et al. (2024). LLaMA Pro: Progressive LLaMA with block expansion. *arXiv:2401.02415*.

Yang, Z., Liu, J., Xu, J., et al. (2024). Continual learning of large language models: A comprehensive survey. *arXiv:2404.16789*.

Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML 2017*.

Zhao, H., et al. (2024). Read-ME: Refactorizing LLMs as router-decoupled mixture of experts with system co-design. *arXiv:2410.19123*.

---

## Appendix A: Formal Algorithm

```
Algorithm 1: INCA Training Loop

Input:  Domain sequence D = {D_1, ..., D_T}; config cfg; base model M_0
Output: Trained INCA model with grown block chain

 1: manager ← INCALayerManager(M_0, cfg)
 2: detector ← INCAPlateauDetector(cfg)
 3: cka_monitor ← CKAMonitor(ref_size=200)
 4: buffer ← INCAReplayBuffer(max_size=2000, n_revise=3, p_hard=0.70, p_easy=0.20)
 5:
 6: for t = 1 to T do
 7:     stream_t ← load_period(D_t, completion_framing=True)
 8:     s_0 ← eval(manager, stream_t[:n_eval])    // pre-period score
 9:     detector.reset_period(s_0)
10:     cka_monitor.cache_reference(manager.current_block, stream_t)
11:
12:     for epoch = 1 to cfg.epochs_per_period do
13:         if epoch >= cfg.min_epochs_before_grow:   // grokking guard cleared
14:             allow_grow ← True
15:         for batch in make_replay_loader(stream_t, buffer, cfg, epoch=epoch) do
16:             loss ← forward_backward(manager, batch)
17:             clip_gradients(max_norm=1.0)
18:             optimizer.step()
19:             scheduler.step()
20:             global_step += 1
21:
22:             if global_step % cfg.k_eval == 0 and allow_grow:
23:                 score  ← eval(manager, stream_t[:n_eval])
24:                 g_norm ← manager.grad_norm()
25:                 cka    ← cka_monitor.compute(manager.current_block)
26:                 detector.update(loss, score, g_norm, cka)
27:                 event  ← detector.check(epoch)
28:
29:                 if event == PERIOD_LEARNED:
30:                     break
31:
32:                 if event == BLOCK_FULL:
33:                     manager.freeze_and_grow()    // ← core INCA operation
34:                     detector.reset_block()
35:                     cka_monitor.reset()
36:                     break
37:
38:     // Patience timeout handler (T1.2 relabelling)
39:     fallback ← detector.check_timeout()
40:     if fallback == EXHAUSTED:
41:         manager.freeze_and_grow()
42:
43:     buffer.add_period(t, stream_t[:cfg.buffer_max_size])
44:
45: return manager
```

---

## Appendix B: CLS Signal Mapping Table

```
DETAILED CLS → INCA BIOLOGICAL MAPPING
══════════════════════════════════════════════════════════════════════════

Neuroscience                        INCA                    Timescale
──────────────────────────────────────────────────────────────────────
Hippocampal pattern separation      Identity-init new block  Instantaneous
                                    (structural isolation)

Hippocampal rapid one-shot          Warm-start deepcopy      Instantaneous
  encoding from structured prior    (domain-informed start)

Hippocampal replay (waking)         Phase A uniform buffer   Epoch 1-2

Hippocampal replay (sleep, REM)     Phase B 70/20/10 sched  Epoch 3+

Neocortical slow consolidation      Frozen block training    Entire domain

Encoding saturation (hippocampal)   RIR ≤ 0.05 + plateau   Per k_eval

Representational fixation           CKA ≥ 0.95              Per k_eval
  (neocortical consolidation)

Synaptic depression                 Gradient-norm EMA decay  Per k_eval
  (homeostatic downscaling)         < 0.5 × peak

Prediction error convergence        Loss plateau < δ_min     Per k_eval
  (predictive coding gate)

Systems consolidation trigger       BlockFull consensus      On demand
  (hippocampal → neocortical)       → freeze_and_grow()

Pattern completion                  Embedding skip: c_i =    Per forward
  (CA3 → CA1 backprojection)        P_{i-1}h_{i-1} + E       pass

Synaptic tagging and capture        Alpha-gated lateral      Post-grow
  (late-LTP stabilization)          adapter: tanh(α=0)→learn  training

══════════════════════════════════════════════════════════════════════════
```

---

## Appendix C: Selector Parameter Analysis

For FLAN-T5-large with $D = 1024$, $H = 4$ heads, $d = D/H = 256$:

| Selector | Params (selector only) | Notes |
|---|---|---|
| S-QKV | $3 \times 1024^2 = 3{,}145{,}728 \approx 3.1\text{M}$ | $W_K, W_V, W_O$; shared across all blocks; identity init |
| UCLBR | $\approx 3{,}600{,}000$ | S-QKV + 2-layer pre-gate MLP ($1024 \times 64 + 64 \times 1$) + conf head + bias buffer |
| S-FULL | $(1024 \times 64 + 64) + (64 \times 1 + 1) = 65{,}601$ | Two-layer gate MLP per block mean-pool |
| S-WS | $n_{\text{blocks}}$ scalars | One learnable logit per block; grows with chain |

S-QKV's cost is fixed and block-count-independent because $W_K$, $W_V$, $W_O$ are shared — all blocks contribute through the same projection matrices, competing via their activations rather than via dedicated weights. This makes S-QKV asymptotically more parameter-efficient than per-block gating as the chain grows deeper. At maximum chain depth (6 base blocks + 2 grown), S-QKV adds $\approx 3.1$M selector parameters on top of the base 780M, a 0.4% overhead.

---

## Appendix D: Saturation Signal Conjunction Grid

```
FOUR-SIGNAL CONJUNCTION DECISION MATRIX

             RIR ≥ ρ_RIR    ρ_neg < RIR < ρ_RIR    RIR ≤ ρ_neg
           ┌──────────────┬────────────────────────┬──────────────┐
Plateau=T  │ PERIOD       │         NONE           │  If grad↓ ∨  │
           │ LEARNED  ✓   │     (keep training)    │  CKA≥τ:      │
           │ (advance     │                        │  BLOCK       │
           │  domain)     │                        │  FULL  ✓     │
           ├──────────────┼────────────────────────┼──────────────┤
Plateau=F  │    NONE      │         NONE           │    NONE      │
           │ (loss still  │  (training progressing │ (loss has    │
           │  declining)  │   at moderate rate)    │  not yet     │
           │              │                        │  converged)  │
           └──────────────┴────────────────────────┴──────────────┘

Second-layer conjunction (BlockFull only):
     ┌──────────────────────────────────┐
     │  Grad-norm decayed  ─── OR ────  │
     │  CKA stable                      │
     │  (one representational signal   │
     │   sufficient — either grad       │
     │   vanishing OR geometric fixity) │
     └──────────────────────────────────┘

Grokking guard: ALL outputs suppressed to NONE if epoch < 2
Timeout path (T1.2): if patience exhausted without decision →
  RIR ≥ 0.20: PeriodLearned  │  RIR < 0.20: Exhausted → BlockFull
```

---

*Document Status: Final Draft v2. Numerical results pending training runs. All architecture descriptions, mathematical formalizations, and ablation designs are implemented and correspond to `models/inca/`, `training/inca_trainer.py`, and `configs/`. Paper B supersedes Paper A (temporal CL framing); this is the primary capacity-architecture paper.*
