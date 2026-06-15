#!/usr/bin/env python3
"""Download a dataset or competition data from Kaggle.

Requires ~/.kaggle/kaggle.json with valid API credentials.
Setup: kaggle.com → Profile → Settings → API → Create New Token
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile


CREDS_PATH = os.path.expanduser("~/.kaggle/kaggle.json")


def ensure_deps():
    try:
        import kaggle  # noqa
    except ImportError:
        print("Installing kaggle...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "kaggle", "--break-system-packages", "-q"]
        )


def check_credentials():
    """Verify kaggle.json exists and is readable. Print setup help if not."""
    if not os.path.exists(CREDS_PATH):
        print("[Auth Required] Kaggle API credentials not found.")
        print()
        print("  Setup steps:")
        print("  1. Go to: https://www.kaggle.com/settings  (Account tab)")
        print("  2. Scroll to 'API' section → click 'Create New Token'")
        print("  3. This downloads kaggle.json — move it:")
        print("       mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("  4. Lock the file:")
        print("       chmod 600 ~/.kaggle/kaggle.json")
        print("  5. Re-run this script.")
        sys.exit(1)

    os.chmod(CREDS_PATH, 0o600)
    with open(CREDS_PATH) as f:
        creds = json.load(f)
    if "username" not in creds or "key" not in creds:
        print(f"[Error] {CREDS_PATH} looks malformed. Re-download it from kaggle.com.")
        sys.exit(1)


def extract_zip(zip_path, dest_dir):
    """Extract a zip file and remove the archive."""
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    os.remove(zip_path)


def download_dataset(slug, output_dir):
    """Download a standalone Kaggle dataset (format: 'owner/dataset-name')."""
    import kaggle

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading Kaggle dataset '{slug}'...")

    try:
        kaggle.api.dataset_download_files(
            slug,
            path=output_dir,
            unzip=True,
            quiet=False,
        )
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "forbidden" in msg or "not found" in msg or "404" in msg:
            print(f"\n[Error] Could not download '{slug}'.")
            print("  Check that the dataset slug is correct: kaggle.com/datasets/<owner>/<name>")
            print("  If it's a private dataset, make sure your account has access.")
        else:
            print(f"[Error] {e}")
        sys.exit(1)

    _report(slug, output_dir)


def download_competition(name, output_dir):
    """Download a Kaggle competition dataset."""
    import kaggle

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading Kaggle competition data '{name}'...")

    try:
        kaggle.api.competition_download_files(
            name,
            path=output_dir,
            quiet=False,
        )
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "forbidden" in msg:
            print(f"\n[Auth Required] You must accept the competition rules first.")
            print(f"  Go to: https://www.kaggle.com/c/{name}/rules  and click 'I Understand and Accept'")
        elif "404" in msg or "not found" in msg:
            print(f"\n[Error] Competition '{name}' not found. Check the name at kaggle.com/competitions.")
        else:
            print(f"[Error] {e}")
        sys.exit(1)

    # Unzip any downloaded zips
    for fname in os.listdir(output_dir):
        if fname.endswith(".zip"):
            extract_zip(os.path.join(output_dir, fname), output_dir)

    _report(name, output_dir)


def _report(name, output_dir):
    abs_path = os.path.abspath(output_dir)
    total = 0
    print(f"\n✓  {name}  →  {abs_path}/")
    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total += sz
            rel = os.path.relpath(fp, abs_path)
            print(f"   {rel}  ({sz / 1024 / 1024:.1f} MB)")
    print(f"   Total: {total / 1024 / 1024:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description="Download a Kaggle dataset or competition")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset slug, e.g. 'Cornell-University/arxiv'")
    group.add_argument("--competition", help="Competition name, e.g. 'titanic'")
    ap.add_argument("--output", required=True, help="Destination directory")
    args = ap.parse_args()

    ensure_deps()
    check_credentials()

    if args.dataset:
        download_dataset(args.dataset, args.output)
    else:
        download_competition(args.competition, args.output)


if __name__ == "__main__":
    main()
