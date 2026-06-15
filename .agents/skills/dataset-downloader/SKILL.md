---
name: dataset-downloader
description: >
  Use this skill whenever the user wants to download a dataset by name or URL — from anywhere.
  Trigger on: "download the X dataset", "get me X dataset", "I need the X dataset",
  "download X to [path]", "fetch the X benchmark", "grab the X corpus", or any message where
  the user names a dataset and wants it on their machine. The user will NOT tell you where the
  dataset lives — you figure that out yourself by searching or using your knowledge. This skill
  covers HuggingFace, Kaggle, GitHub/Git, direct URLs, and academic repositories. Use it
  proactively: if the user names a dataset anywhere in conversation, offer to download it.
---

# Dataset Downloader

The user gives you a dataset name (and optionally a destination path). You find it, brief
the user, get confirmation, then download it. Download only — no loading, no notebooks.

---

## Phase 1 — Discover the Dataset

Use your knowledge first. Many major datasets have well-known canonical homes:
- NLP/vision/audio benchmarks → HuggingFace Hub (`huggingface.co/datasets`)
- Competition datasets → Kaggle
- Research paper datasets → GitHub repo linked from the paper, or Zenodo/OpenReview
- Web-crawl corpora (CC-News, Common Crawl, C4, The Pile) → HuggingFace or official site

If you're not confident, **search the web**. Good queries:
- `"[name] dataset" site:huggingface.co`
- `"[name] dataset" site:kaggle.com`
- `"[name] dataset" download github`
- `"[name] dataset" paperswithcode`

From search results and your knowledge, identify the **single best source** using this
preference order:
1. **HuggingFace Hub** — best API, handles resumption, widest ML dataset coverage
2. **Kaggle** — best for competition data and tabular datasets
3. **GitHub / Git** — for paper repos, academic datasets hosted on git, small-to-medium data
4. **Direct URL** — official project pages, Zenodo, academic servers, any raw file link

If after searching you genuinely cannot find a reliable download source, stop and tell the
user clearly: what you searched for, what you found (if anything), and what they could try
manually. Don't guess or make up a download path.

---

## Phase 2 — Pre-Download Brief

Before downloading anything, present a short summary to the user:

```
Dataset:  [full name + brief description — domain, task, what's in it]
Source:   [HuggingFace / Kaggle / GitHub / URL — with the exact path or URL]
Format:   [CSV, Parquet, JSON, images, audio, etc.]
Size:     [estimated total size if known]
License:  [license if known]
Auth:     [none required / HF token needed / Kaggle API key needed]
```

Then, **if the estimated size is large (> ~5 GB)**, ask the user before proceeding:
> "This dataset is approximately [size]. It has the following subsets/splits: [list].
> Do you want the full dataset, or a specific split/subset/date range?"

For small datasets, skip the size question and just ask: "Ready to download to [path]?"

Wait for the user to confirm (or adjust) before running anything.

---

## Phase 3 — Auth Setup (if needed)

If the source requires credentials, walk the user through setup before attempting the download.

**HuggingFace token** (required for gated/private datasets):
> "This dataset requires a HuggingFace token.
> 1. Go to https://huggingface.co/settings/tokens
> 2. Create a new token (read access is enough)
> 3. Either set it as: `export HF_TOKEN=hf_your_token`
>    Or pass it with `--token hf_your_token` when running the script"

**Kaggle API key**:
> "This dataset requires a Kaggle API key.
> 1. Go to kaggle.com → Profile → Settings → API → Create New API Token
> 2. This downloads `kaggle.json` — move it to `~/.kaggle/kaggle.json`
> 3. Run: `chmod 600 ~/.kaggle/kaggle.json`"

**GitHub token** (private repos only):
> "Set `export GITHUB_TOKEN=ghp_your_token` or pass `--token` to the script."

---

## Phase 4 — Download

Run the appropriate script from `scripts/`. The exact command depends on the source:

### HuggingFace
```bash
python scripts/download_hf.py \
  --dataset "username/dataset-name" \
  --output "/path/to/save" \
  [--split "train,test,validation"] \
  [--config "config-name"] \
  [--token "hf_..."]
```
Read `references/huggingface.md` for how to handle configs, streaming, and gated datasets.

### Kaggle
```bash
python scripts/download_kaggle.py \
  --dataset "username/dataset-slug" \
  --output "/path/to/save"
# For competition data:
python scripts/download_kaggle.py \
  --competition "competition-name" \
  --output "/path/to/save"
```
Read `references/kaggle.md` for credential setup and dataset vs competition syntax.

### GitHub / Git
```bash
python scripts/download_git.py \
  --repo "https://github.com/username/repo" \
  --output "/path/to/save" \
  [--path "subfolder/within/repo"] \
  [--branch "main"] \
  [--tag "v1.0"] \
  [--release-asset "filename.zip"] \
  [--token "ghp_..."]
```
Read `references/github.md` for sparse checkout vs full clone vs release asset strategies.

### Direct URL / Zenodo / Academic Sites
```bash
python scripts/download_url.py \
  --url "https://example.com/dataset.zip" \
  --output "/path/to/save" \
  [--filename "custom-name.zip"] \
  [--no-extract]
```

---

## Phase 5 — Verify and Report

After the download completes, show the user:
1. The absolute path where data landed
2. Directory listing with file sizes (`ls -lh` output or equivalent)
3. Total downloaded size
4. A confirmation that the download looks complete (no obvious truncation)

Example:
```
✓  imdb  →  /home/user/data/imdb/
   train/  (25,000 examples, ~12 MB)
   test/   (25,000 examples, ~12 MB)
   unsupervised/  (50,000 examples, ~24 MB)
   Total: 48 MB
```

If anything looks wrong (zero-byte files, partial archives, error messages in output),
tell the user what went wrong and what to try next.
