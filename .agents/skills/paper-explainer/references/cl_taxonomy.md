# Continual learning — a one-page field map

Use this to anchor the **CL Connection** section. The user wants broad-CL
framing, not project-specific tie-ins, so refer to the literature here and
not to any specific implementation in the user's own work.

## The seven families

Most of CL fits into one of these seven buckets. When you place a paper, name
the bucket and one or two close neighbours, then say what trade-off this paper
makes that the neighbours don't.

### 1. Regularisation-based
Add a penalty term to the loss that discourages moving parameters important
to old tasks.

- **EWC** — Kirkpatrick et al., PNAS 2017. Diagonal Fisher penalty.
- **SI** — Zenke et al., ICML 2017. Path-integral importance.
- **MAS** — Aljundi et al., ECCV 2018. Output-sensitivity importance.
- **OWM** — Zeng et al., NMI 2019. Project gradients orthogonal to old input
  subspace; recovers some of what EWC's diagonal Fisher discards.

Trade-off: cheap (no extra memory), but the importance estimate is only as
good as the Fisher/path/sensitivity proxy — fails on long task sequences and
when tasks share features in non-obvious ways.

### 2. Replay-based
Keep a small buffer of old data (or generate it) and mix it into training.

- **ER** — Chaudhry et al., ICLR 2019 W. Plain reservoir replay.
- **GEM** — Lopez-Paz & Ranzato, NeurIPS 2017. Constrain gradient to not
  increase loss on a memory of old samples.
- **A-GEM** — Chaudhry et al., ICLR 2019. Cheaper average-gradient version.
- **MIR** — Aljundi et al., NeurIPS 2019. Maximally Interfered Retrieval —
  replay only the samples that the current update would hurt most.
- **DER, DER++** — Buzzega et al., NeurIPS 2020. Replay logits, not just
  labels.
- **GDumb** — Prabhu et al., ECCV 2020. The infamous "just train on the
  buffer at eval time" baseline that beats most CL methods.

Trade-off: by far the most reliable family on benchmarks, but requires
storing data (privacy/regulatory issues) and the buffer-vs-stream ratio
tuning is delicate.

### 3. Parameter-isolation / dynamic-architecture
Different parameters for different tasks; either grow the network or mask
out per-task subsets.

- **PNN (Progressive Neural Networks)** — Rusu et al., 2016. Add a column
  per task; freeze old columns.
- **PackNet** — Mallya & Lazebnik, CVPR 2018. Iteratively prune-then-train
  to pack many tasks into one network.
- **HAT (Hard Attention to the Task)** — Serra et al., ICML 2018. Learnable
  per-task attention masks over units.
- **SupSup** — Wortsman et al., NeurIPS 2020. Supermasks per task on top of
  a frozen random net.
- **DEN** — Yoon et al., ICLR 2018. Dynamically expand only when needed.

Trade-off: zero forgetting by construction, but parameter cost grows with
tasks (PNN), and routing the right task at inference is its own problem.

### 4. Prompt-based (vision-language / foundation-model era)
Freeze a large pre-trained backbone; learn small per-task prompts that
condition it.

- **L2P (Learning to Prompt)** — Wang et al., CVPR 2022.
- **DualPrompt** — Wang et al., ECCV 2022. General + expert prompts.
- **CODA-Prompt** — Smith et al., CVPR 2023. Continual decomposed
  attention-based prompting.
- **S-Prompts** — Wang et al., NeurIPS 2022. Domain-specific prompts.

Trade-off: very cheap per task, leverages pre-training, but only works when
you have a strong frozen backbone — collapses to nothing on small models or
out-of-distribution tasks.

### 5. Distillation / rehearsal-free knowledge transfer
Use the old model itself as a soft teacher when training on new data.

- **LwF (Learning without Forgetting)** — Li & Hoiem, TPAMI 2017.
- **iCaRL** — Rebuffi et al., CVPR 2017. Distillation + small exemplar set.
- **PODNet** — Douillard et al., ECCV 2020. Spatial-pooled distillation.
- **LUCIR** — Hou et al., CVPR 2019. Cosine-norm classifier + less-forget
  constraint.

Trade-off: doesn't need a large memory, but the soft teacher's bias compounds
across tasks.

### 6. Modular / mixture-of-experts
Compose specialised modules (LoRA adapters, MoE experts, additional blocks)
per task or stage.

- **LoRA-MoE** lineage — multiple variants combining low-rank adapters with
  a router.
- **SEED (Selective Expansion via Expert Distillation)** — Rypeść et al.,
  ICLR 2024.
- **Progressive Prompts** — Razdaibiedina et al., ICLR 2023. Concatenate
  per-task prompts.

Trade-off: clean separation between tasks, but the router is the bottleneck —
in many regimes a Mixture-of-LoRAs collapses to a single expert.

### 7. Test-time / online adaptation
Adapt the model at inference time without persistent updates.

- **TENT** — Wang et al., ICLR 2021. Entropy-minimisation BN updates at
  test time.
- **CoTTA** — Wang et al., CVPR 2022. Continual test-time adaptation with
  weight averaging.
- **EATA** — Niu et al., ICML 2022. Efficient test-time adaptation.

Trade-off: doesn't touch the training pipeline, but only works for
distribution shift, not task addition.

## Cross-cutting concepts to namedrop where relevant

- **Stability–plasticity dilemma** — the headline tension; every CL method
  is a point on this curve.
- **Forward / backward transfer** — does learning task t help task t+1
  (FWT) or hurt task t-1 (negative BWT)?
- **CKA (Centred Kernel Alignment)** — Kornblith et al., ICML 2019. The
  standard tool for measuring representation drift across tasks.
- **Plasticity loss / dormant neurons** — Lyle et al., ICML 2023; Sokar et
  al., 2023. The phenomenon that adapted networks lose the ability to
  learn new things even when not catastrophically forgetting.
- **Linear probing vs. full fine-tuning evaluation** — Davari et al., 2022.
  CL methods that look great under linear probing often collapse under full
  fine-tuning, and vice versa.

## Standard surveys and benchmarks to cite

- de Lange et al., TPAMI 2021 — "A continual learning survey: defying
  forgetting in classification tasks." The canonical taxonomy reference.
- Wang et al., 2023 — "A comprehensive survey of continual learning:
  theory, method and application." More recent, broader.
- Mehta et al., 2023 — "An empirical investigation of the role of
  pre-training in lifelong learning."
- Lesort et al., 2020 — "Continual learning for robotics."
- Benchmarks: Split-CIFAR-{10,100}, Permuted-MNIST, CORe50, TiC-CLIP,
  TemporalWiki (LM continual learning), TRACE (instruction-following CL).

## Useful lenses when there's no obvious connection

When a paper isn't directly about CL, look for one of these angles:

- **Optimisation papers** → connection via plasticity / loss-landscape /
  regularisation effect on forgetting.
- **Representation-learning papers** → connection via feature drift, CKA,
  and what makes representations transferable across distributions.
- **Architecture papers** → connection via parameter efficiency and
  capacity-per-task, which directly bears on parameter-isolation methods.
- **RLHF / fine-tuning papers** → connection via alignment-tax and
  catastrophic forgetting of pre-training abilities under instruction tuning.
- **Theory papers (PAC-Bayes, NTK, etc.)** → connection via generalisation
  bounds across task distributions.

Use these only when honest. A vision transformer paper with no CL angle gets
"this paper has no direct CL contribution; its [X] would be a useful building
block for [Y class of CL method] because [reason]" — a single sentence, not
a forced section.
