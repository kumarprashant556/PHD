#!/usr/bin/env python3
"""Download a dataset from HuggingFace Hub.

Tries the `datasets` library first (structured datasets).
Falls back to `huggingface_hub` snapshot download (raw files) if that fails.
"""

import argparse
import os
import subprocess
import sys


def ensure_deps():
    for pkg in ["datasets", "huggingface_hub"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"]
            )


def download_structured(dataset_id, output_dir, splits, config, token):
    """Download via the `datasets` library (structured, schema-aware)."""
    from datasets import load_dataset, get_dataset_config_names, DatasetDict

    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect config if not specified
    if config is None:
        try:
            configs = get_dataset_config_names(dataset_id, token=token)
            if configs and configs != ["default"]:
                print(f"  Available configs: {configs}")
                config = configs[0]
                print(f"  Using: {config}")
        except Exception:
            pass

    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    if config:
        kwargs["name"] = config

    split_list = [s.strip() for s in splits.split(",")] if splits else None

    print(f"Fetching '{dataset_id}'{f' [{config}]' if config else ''}...")

    try:
        if split_list:
            ds_dict = {}
            for s in split_list:
                print(f"  split: {s}")
                ds_dict[s] = load_dataset(dataset_id, split=s, **kwargs)
            dataset = DatasetDict(ds_dict)
        else:
            dataset = load_dataset(dataset_id, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "gated" in msg or "403" in msg or "unauthorized" in msg or "access" in msg:
            _gated_message(dataset_id)
            sys.exit(1)
        # Not a structured dataset — let caller fall back
        return False

    print(f"Saving to {output_dir} ...")
    dataset.save_to_disk(output_dir)

    print(f"\n✓  {dataset_id}  →  {os.path.abspath(output_dir)}/")
    if hasattr(dataset, "items"):
        for name, ds in dataset.items():
            print(f"   {name}/  ({len(ds):,} rows, columns: {list(ds.features.keys())})")
    else:
        print(f"   {len(dataset):,} rows  |  columns: {list(dataset.features.keys())}")

    print(f"\n   Load with:")
    print(f"     from datasets import load_from_disk")
    print(f"     ds = load_from_disk('{os.path.abspath(output_dir)}')")
    return True


def download_snapshot(dataset_id, output_dir, token):
    """Download raw files via huggingface_hub.snapshot_download."""
    from huggingface_hub import snapshot_download

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading raw files for '{dataset_id}'...")

    try:
        local = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=output_dir,
            token=token,
        )
    except Exception as e:
        msg = str(e).lower()
        if "gated" in msg or "403" in msg or "unauthorized" in msg or "access" in msg:
            _gated_message(dataset_id)
        else:
            print(f"[Error] {e}")
        sys.exit(1)

    print(f"\n✓  {dataset_id}  →  {os.path.abspath(local)}/")
    total = 0
    for root, dirs, files in os.walk(local):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total += sz
            rel = os.path.relpath(fp, local)
            print(f"   {rel}  ({sz / 1024 / 1024:.1f} MB)")
    print(f"   Total: {total / 1024 / 1024:.1f} MB")


def _gated_message(dataset_id):
    print(f"\n[Auth Required] '{dataset_id}' is a gated dataset.")
    print(f"  1. Go to: https://huggingface.co/datasets/{dataset_id}")
    print(f"  2. Accept the access terms on that page.")
    print(f"  3. Get a token: https://huggingface.co/settings/tokens")
    print(f"  4. Re-run with:  --token hf_your_token_here")
    print(f"     or set:        export HF_TOKEN=hf_your_token_here")


def main():
    ap = argparse.ArgumentParser(description="Download a HuggingFace dataset")
    ap.add_argument("--dataset", required=True, help="e.g. 'imdb' or 'allenai/c4'")
    ap.add_argument("--output", required=True, help="Destination directory")
    ap.add_argument("--split", default=None, help="Comma-separated splits: 'train,test'")
    ap.add_argument("--config", default=None, help="Dataset config/subset name")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF API token")
    ap.add_argument("--files-only", action="store_true", help="Skip datasets lib, download raw files")
    args = ap.parse_args()

    ensure_deps()

    if args.files_only:
        download_snapshot(args.dataset, args.output, args.token)
    else:
        ok = download_structured(args.dataset, args.output, args.split, args.config, args.token)
        if not ok:
            print("Not a structured dataset — downloading raw files...")
            download_snapshot(args.dataset, args.output, args.token)


if __name__ == "__main__":
    main()
