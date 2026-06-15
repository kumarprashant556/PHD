"""Entry point: launch an ablation sweep.

Usage
-----
python scripts/run_ablation.py --config configs/ablations/e_route.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    sweep = cfg.get("sweep", {})
    # Build grid from all sweep keys
    import itertools
    keys   = list(sweep.keys())
    values = list(sweep.values())
    grid   = list(itertools.product(*values))

    base_config = cfg.get("defaults", ["configs/inca"])[0].lstrip("/") + ".yaml"
    print(f"Sweep: {len(grid)} runs over {keys}")

    for combo in grid:
        overrides = " ".join(f"--{k} {v}" for k, v in zip(keys, combo))
        cmd = f"python scripts/train_inca.py --config {base_config} {overrides}"
        print(f"  Running: {cmd}")
        if not args.dry_run:
            subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
