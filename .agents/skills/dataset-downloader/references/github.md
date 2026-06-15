# GitHub / Git Dataset Reference

## Choosing the right strategy

| Situation | Strategy | Flag |
|---|---|---|
| Only need a subfolder of a large repo | Sparse checkout | `--path folder/name` |
| Need the whole repo, don't care about history | Shallow clone | _(default, no extra flags)_ |
| Dataset is in a GitHub Release as a zip/tar | Release asset | `--release-asset filename.zip` |
| Specific branch or tag | Any of the above | `--branch name` or `--tag v1.0` |

## Sparse checkout example

For a repo like `github.com/google-research/google-research` where you only need
the `fruit_detection/data/` subfolder:

```bash
python scripts/download_git.py \
  --repo https://github.com/google-research/google-research \
  --path fruit_detection/data \
  --output ./data/fruit-detection
```

This avoids cloning the entire (large) repo.

## Release asset example

When a dataset is published as a release attachment:
```bash
python scripts/download_git.py \
  --repo https://github.com/some-org/some-dataset \
  --release-asset dataset_v1.zip \
  --output ./data/some-dataset
```

If `--release-asset` is given without a filename, the first asset is downloaded.

## Authentication (private repos / rate limits)

Set `GITHUB_TOKEN` or pass `--token ghp_your_token`:
- Fine-grained token with "Contents: Read" permission is enough
- Also avoids GitHub's 60 req/hr unauthenticated API rate limit

## Size considerations

- Shallow clone (`--depth=1`) skips all git history, getting only the latest snapshot
- For repos with large files tracked via Git LFS, install `git-lfs` first:
  `git lfs install` then `git lfs pull` after cloning
- If the repo uses Git LFS and you get pointer files instead of actual data,
  run: `git lfs pull` inside the cloned directory
