# HuggingFace Dataset Reference

## Finding the right dataset ID and config

The dataset ID is the `owner/name` slug from the HuggingFace URL.
Example: `https://huggingface.co/datasets/allenai/c4` → ID is `allenai/c4`

Some datasets have multiple **configs** (subsets):
- `allenai/c4` has configs: `en`, `en.noblocklist`, `realnewslike`, etc.
- `facebook/flores` has one config per language pair
- If no config is given, the script auto-detects the first one

To list all available configs for a dataset:
```python
from datasets import get_dataset_config_names
print(get_dataset_config_names("allenai/c4"))
```

## Common large datasets and tips

| Dataset | ID | Notes |
|---|---|---|
| CC-News | `cc_news` | ~74k articles; manageable size |
| C4 | `allenai/c4` | Config `en` is ~300GB — use streaming or a split |
| The Pile | `EleutherAI/pile` | ~800GB — always ask user for subset |
| Common Crawl | Not on HF directly | Use `allenai/c4` or `bigscience/roots` subsets |
| Wikipedia | `wikimedia/wikipedia` | Config = language code, e.g. `20231101.en` |
| IMDB | `imdb` | Small, no config needed |
| CIFAR-10 | `uoft-cs/cifar10` | Image dataset |
| ImageNet | `ILSVRC/imagenet-1k` | Gated — requires HF token + terms acceptance |

## Streaming large datasets (alternative to full download)

For datasets > 50GB, Claude should mention streaming as an alternative:
```python
from datasets import load_dataset
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
for example in ds.take(1000):
    print(example)
```

## Gated datasets

Gated datasets require:
1. User visits the dataset page and accepts terms
2. User creates a HF token at huggingface.co/settings/tokens
3. Token passed via `--token` or `HF_TOKEN` env var

Common gated datasets: Llama (model, not dataset), ImageNet, some medical datasets.

## Token scopes

Read-only token is sufficient for all public and gated dataset downloads.
Write tokens are only needed for uploading.
