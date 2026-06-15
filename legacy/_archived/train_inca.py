"""Entry point: train INCA.

Usage
-----
# From repo root:
python scripts/train_inca.py --config configs/inca.yaml
python scripts/train_inca.py --config configs/inca.yaml --dataset cc_news --selector embedding_query --seed 42
python scripts/train_inca.py --config configs/inca.yaml --dry-run
python scripts/train_inca.py --config configs/ablations/e_route.yaml --selector uclbr
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from models.inca.config import INCAConfig
from training.inca_trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train INCA continual learning model")
    p.add_argument("--config",   type=str, default="configs/inca.yaml",
                   help="Path to YAML config (relative to repo root)")
    p.add_argument("--dataset",  type=str, default=None,
                   help="Override cfg.dataset (cc_news|redpajama|tic_lm|realtimeqa|temporalwiki|trace)")
    p.add_argument("--selector", type=str, default=None,
                   help="Override cfg.selector (embedding_query|uclbr|cross_attention|weighted_sum)")
    p.add_argument("--seed",     type=int, default=None, help="Override cfg.seed")
    p.add_argument("--device",   type=str, default=None, help="Override device (cpu|mps|cuda)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Build config + model only — do not train")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}

    # CLI overrides (only apply if explicitly passed)
    if args.dataset:  cfg_dict["dataset"]   = args.dataset
    if args.selector: cfg_dict["selector"]  = args.selector
    if args.seed:     cfg_dict["seed"]      = args.seed
    if args.device:   cfg_dict["device"]    = args.device   # inca_trainer reads via getattr

    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items()
                        if k in INCAConfig.__dataclass_fields__})

    print(f"INCA  |  dataset={cfg.dataset}  selector={cfg.selector}  "
          f"seed={cfg.seed}  model={cfg.model_name}")

    if args.dry_run:
        print("--dry-run: config valid, exiting.")
        return

    train(cfg)


if __name__ == "__main__":
    main()
