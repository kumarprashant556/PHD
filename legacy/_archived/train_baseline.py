"""Entry point: train baseline FLAN-T5 (sequential or joint).

Usage
-----
    # Sequential baseline (catastrophic forgetting floor)
    python scripts/train_baseline.py --config configs/inca.yaml --mode sequential

    # Joint baseline (upper bound — all periods merged)
    python scripts/train_baseline.py --config configs/inca.yaml --mode joint

    # Override dataset or seed
    python scripts/train_baseline.py --config configs/inca.yaml \\
        --dataset cc_news --mode sequential --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from models.inca.config import INCAConfig
from training.baseline_trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train FLAN-T5 baseline")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--mode",     default="sequential",
                   choices=["sequential", "joint"],
                   help="sequential (default) or joint")
    p.add_argument("--dataset",  default=None, help="Override cfg.dataset")
    p.add_argument("--seed",     type=int, default=None, help="Override cfg.seed")
    p.add_argument("--device",   default=None, help="cpu | mps | cuda")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}

    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed

    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items()
                        if k in INCAConfig.__dataclass_fields__})

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Baseline  |  mode={args.mode}  |  model={cfg.model_name}  "
          f"|  dataset={getattr(cfg, 'dataset', 'cc_news')}  |  device={device}")

    train(cfg, device, mode=args.mode)


if __name__ == "__main__":
    main()
