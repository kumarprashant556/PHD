# Kaggle Dataset Reference

## Credential setup

Credentials live in `~/.kaggle/kaggle.json`:
```json
{"username": "your_username", "key": "your_api_key"}
```

Setup steps:
1. Go to https://www.kaggle.com/settings (Account tab)
2. Scroll to "API" → click "Create New Token"
3. `kaggle.json` is downloaded automatically
4. `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
5. `chmod 600 ~/.kaggle/kaggle.json`

## Dataset vs Competition syntax

**Standalone datasets** (kaggle.com/datasets/...):
- Slug format: `owner/dataset-name`
- Example: `Cornell-University/arxiv`, `rohanrao/formula-1-world-championship-1950-2020`
- Command: `--dataset owner/dataset-name`

**Competition datasets** (kaggle.com/competitions/...):
- Just the competition name
- Example: `titanic`, `house-prices-advanced-regression-techniques`
- Command: `--competition competition-name`
- Note: user must accept competition rules at kaggle.com/c/<name>/rules first

## Finding the slug

From the Kaggle URL:
- `https://www.kaggle.com/datasets/Cornell-University/arxiv` → `Cornell-University/arxiv`
- `https://www.kaggle.com/c/titanic` → competition name is `titanic`

## Common datasets

| Dataset | Type | Slug |
|---|---|---|
| Titanic | Competition | `--competition titanic` |
| House Prices | Competition | `--competition house-prices-advanced-regression-techniques` |
| ArXiv | Dataset | `--dataset Cornell-University/arxiv` |
| NYC Taxi | Dataset | `--dataset elemento/nyc-yellow-taxi-trip-data` |
| Chest X-ray | Dataset | `--dataset paultimothymooney/chest-xray-pneumonia` |
