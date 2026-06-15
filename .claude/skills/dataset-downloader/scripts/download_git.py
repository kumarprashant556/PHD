#!/usr/bin/env python3
"""Download a dataset hosted on GitHub or any Git repository.

Strategies (auto-selected based on what's requested):
  1. Sparse checkout  — only a specific subfolder of a large repo
  2. Shallow clone    — full repo, depth=1 (no history)
  3. Release asset    — download a specific file from GitHub Releases
  4. Raw file         — single file via raw.githubusercontent.com or direct URL

Requires: git, curl/wget (all standard on Linux/macOS)
Optional: GITHUB_TOKEN env var for private repos or to avoid rate limits
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request


def run(cmd, check=True, capture=False):
    kwargs = {"check": check}
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    return subprocess.run(cmd, **kwargs)


def has_git():
    return shutil.which("git") is not None


def inject_token(url, token):
    """Inject a GitHub PAT into an HTTPS URL."""
    if token and "github.com" in url and url.startswith("https://"):
        return url.replace("https://", f"https://{token}@")
    return url


def sparse_checkout(repo_url, subfolder, output_dir, branch, token):
    """Clone only `subfolder` from `repo_url` using git sparse-checkout."""
    repo_url = inject_token(repo_url, token)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Sparse-checking out '{subfolder}' from {repo_url} ...")
    cmds = [
        ["git", "init"],
        ["git", "remote", "add", "origin", repo_url],
        ["git", "config", "core.sparseCheckout", "true"],
    ]
    for cmd in cmds:
        run(cmd, check=True)

    sparse_file = os.path.join(".git", "info", "sparse-checkout")
    os.makedirs(os.path.dirname(sparse_file), exist_ok=True)
    with open(sparse_file, "w") as f:
        f.write(subfolder.rstrip("/") + "/\n")

    run(["git", "pull", "--depth=1", "origin", branch or "HEAD"])

    # Move the subfolder to output_dir
    src = os.path.join(os.getcwd(), subfolder)
    if os.path.exists(src):
        if os.path.abspath(src) != os.path.abspath(output_dir):
            shutil.copytree(src, output_dir, dirs_exist_ok=True)
    else:
        print(f"[Warning] Subfolder '{subfolder}' not found in repo after checkout.")


def shallow_clone(repo_url, output_dir, branch, token):
    """Shallow clone (depth=1) the full repo."""
    repo_url = inject_token(repo_url, token)
    cmd = ["git", "clone", "--depth=1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, output_dir]
    print(f"Cloning {repo_url} (depth=1)...")
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        print("[Error] git clone failed. Check the URL, branch name, or your credentials.")
        sys.exit(1)


def download_release_asset(owner_repo, asset_name, output_dir, token):
    """Download a specific asset from the latest GitHub Release."""
    import json, urllib.request, urllib.error

    os.makedirs(output_dir, exist_ok=True)
    api_url = f"https://api.github.com/repos/{owner_repo}/releases/latest"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"Fetching release info for {owner_repo}...")
    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            release = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"[Error] GitHub API: {e.code} — check repo name and token.")
        sys.exit(1)

    assets = release.get("assets", [])
    if asset_name:
        matches = [a for a in assets if a["name"] == asset_name]
        if not matches:
            names = [a["name"] for a in assets]
            print(f"[Error] Asset '{asset_name}' not found. Available: {names}")
            sys.exit(1)
        asset = matches[0]
    elif assets:
        asset = assets[0]
        print(f"No --release-asset specified. Downloading first asset: {asset['name']}")
    else:
        print(f"[Error] No assets in the latest release of {owner_repo}.")
        sys.exit(1)

    dest = os.path.join(output_dir, asset["name"])
    print(f"Downloading {asset['name']} ({asset['size'] / 1024 / 1024:.1f} MB)...")

    dl_req = urllib.request.Request(
        asset["browser_download_url"],
        headers={**headers, "Accept": "application/octet-stream"},
    )
    with urllib.request.urlopen(dl_req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)

    _maybe_extract(dest, output_dir)


def _maybe_extract(filepath, output_dir):
    """Extract zip/tar archives in place."""
    import tarfile, zipfile
    if zipfile.is_zipfile(filepath):
        print(f"Extracting {os.path.basename(filepath)}...")
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(output_dir)
        os.remove(filepath)
    elif tarfile.is_tarfile(filepath):
        print(f"Extracting {os.path.basename(filepath)}...")
        with tarfile.open(filepath, "r:*") as t:
            t.extractall(output_dir)
        os.remove(filepath)


def _report(output_dir):
    abs_path = os.path.abspath(output_dir)
    total = 0
    print(f"\n✓  Downloaded  →  {abs_path}/")
    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if not d.startswith(".git")]
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total += sz
            rel = os.path.relpath(fp, abs_path)
            print(f"   {rel}  ({sz / 1024 / 1024:.2f} MB)")
    print(f"   Total: {total / 1024 / 1024:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description="Download a dataset from a Git/GitHub repo")
    ap.add_argument("--repo", required=True, help="Repo URL, e.g. https://github.com/owner/name")
    ap.add_argument("--output", required=True, help="Destination directory")
    ap.add_argument("--path", default=None, help="Subfolder within the repo to download")
    ap.add_argument("--branch", default=None, help="Branch or tag name")
    ap.add_argument("--tag", default=None, help="Tag name (alias for --branch)")
    ap.add_argument("--release-asset", default=None, help="Filename of a GitHub Release asset")
    ap.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    args = ap.parse_args()

    if not has_git():
        print("[Error] git is not installed. Install it and retry.")
        sys.exit(1)

    branch = args.tag or args.branch
    repo_url = args.repo

    # Extract owner/repo for API calls
    owner_repo = None
    if "github.com" in repo_url:
        parts = repo_url.rstrip("/").replace(".git", "").split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            owner_repo = f"{parts[0]}/{parts[1]}"

    if args.release_asset is not None:
        if not owner_repo:
            print("[Error] --release-asset only works with github.com URLs.")
            sys.exit(1)
        download_release_asset(owner_repo, args.release_asset, args.output, args.token)

    elif args.path:
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            sparse_checkout(repo_url, args.path, args.output, branch, args.token)
            os.chdir(orig)

    else:
        shallow_clone(repo_url, args.output, branch, args.token)

    _report(args.output)


if __name__ == "__main__":
    main()
