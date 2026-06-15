#!/usr/bin/env python3
"""Download a dataset from any direct URL.

Handles: zip, tar.gz, gz, plain files, Google Drive shared links.
Uses wget if available (better resume support), falls back to Python urllib.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile


def has_cmd(name):
    return shutil.which(name) is not None


def gdrive_direct_url(url):
    """Convert a Google Drive share link to a direct download URL."""
    # Handle /file/d/<id>/view and /open?id=<id> formats
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        fid = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}&confirm=t"
    return url


def download_with_wget(url, dest_path):
    """Use wget for download — supports resume (-c) and shows progress."""
    cmd = ["wget", "-c", "--show-progress", "-O", dest_path, url]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[Error] wget failed.")
        sys.exit(1)


def download_with_curl(url, dest_path):
    """Use curl as fallback."""
    cmd = ["curl", "-L", "--progress-bar", "-o", dest_path, url]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[Error] curl failed.")
        sys.exit(1)


def download_with_python(url, dest_path):
    """Pure Python fallback — no resume, but always works."""
    print(f"Downloading {url}")
    headers = {"User-Agent": "Mozilla/5.0 dataset-downloader/1.0"}

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 256  # 256 KB
            with open(dest_path, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024 / 1024
                        print(f"\r  {mb:.1f} MB / {total/1024/1024:.1f} MB  ({pct:.0f}%)", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


def extract(filepath, output_dir):
    """Extract archive files. Removes the archive after extraction."""
    name = os.path.basename(filepath).lower()

    if name.endswith(".zip"):
        print(f"Extracting zip...")
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(output_dir)
        os.remove(filepath)

    elif any(name.endswith(ext) for ext in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        print(f"Extracting tar archive...")
        with tarfile.open(filepath, "r:*") as t:
            t.extractall(output_dir)
        os.remove(filepath)

    elif name.endswith(".gz") and not name.endswith(".tar.gz"):
        import gzip
        out_name = os.path.splitext(filepath)[0]
        print(f"Decompressing .gz → {os.path.basename(out_name)}")
        with gzip.open(filepath, "rb") as gz, open(out_name, "wb") as f_out:
            shutil.copyfileobj(gz, f_out)
        os.remove(filepath)

    else:
        print(f"File saved (no extraction needed): {os.path.basename(filepath)}")


def _report(output_dir):
    abs_path = os.path.abspath(output_dir)
    total = 0
    print(f"\n✓  Downloaded  →  {abs_path}/")
    for root, dirs, files in os.walk(abs_path):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total += sz
            rel = os.path.relpath(fp, abs_path)
            print(f"   {rel}  ({sz / 1024 / 1024:.2f} MB)")
    print(f"   Total: {total / 1024 / 1024:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description="Download a dataset from a direct URL")
    ap.add_argument("--url", required=True, help="Download URL")
    ap.add_argument("--output", required=True, help="Destination directory")
    ap.add_argument("--filename", default=None, help="Override filename (default: inferred from URL)")
    ap.add_argument("--no-extract", action="store_true", help="Keep archive as-is, don't extract")
    ap.add_argument("--gdrive", action="store_true", help="Treat URL as a Google Drive share link")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    url = args.url
    if args.gdrive or "drive.google.com" in url:
        url = gdrive_direct_url(url)
        print(f"Google Drive → direct URL: {url}")

    # Infer filename
    if args.filename:
        filename = args.filename
    else:
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path) or "dataset"
        if not filename or filename in (".", "/"):
            filename = "dataset.bin"

    dest_path = os.path.join(args.output, filename)

    print(f"Destination: {os.path.abspath(dest_path)}")

    if has_cmd("wget"):
        download_with_wget(url, dest_path)
    elif has_cmd("curl"):
        download_with_curl(url, dest_path)
    else:
        download_with_python(url, dest_path)

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        print("[Error] Download produced an empty or missing file.")
        sys.exit(1)

    if not args.no_extract:
        extract(dest_path, args.output)

    _report(args.output)


if __name__ == "__main__":
    main()
