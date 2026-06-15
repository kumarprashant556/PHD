"""Evaluation harness  (capsel/evaluation/eval_runner.py)

Load a checkpoint, run on all periods, compute full metric suite,
write JSON report.

Usage
-----
from evaluation.eval_runner import evaluate_checkpoint

results = evaluate_checkpoint(
    checkpoint_path="results/inca_run_001/inca_final.pt",
    period_loaders=period_loaders,   # dict: period -> DataLoader
    cfg=cfg,
)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import torch


def evaluate_checkpoint(
    checkpoint_path: str,
    period_loaders: Dict,
    cfg,
    device: Optional[torch.device] = None,
) -> dict:
    """Load checkpoint and compute BWT/ACC/FWT/RIR across all periods.

    Returns a results dict that is also written to
    {checkpoint_dir}/eval_results.json.
    """
    from models.inca.layer_manager import INCALayerManager
    from evaluation.metrics import compute_metrics
    from transformers import AutoModelForSeq2SeqLM

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    manager    = INCALayerManager(base_model, cfg).to(device)
    manager.load_manager_state(ckpt["manager_state"])
    manager.eval()

    # Collect per-period losses
    period_losses: Dict[str, float] = {}
    for period, loader in period_loaders.items():
        total_loss = n_batches = 0
        with torch.no_grad():
            for batch in loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                lbls  = batch["labels"].to(device)
                enc   = manager(ids, attention_mask=mask)
                out   = base_model(
                    encoder_outputs=(enc,),
                    attention_mask=mask,
                    labels=lbls,
                )
                total_loss += out.loss.item()
                n_batches  += 1
        period_losses[period] = total_loss / max(1, n_batches)

    metrics = compute_metrics(period_losses)
    results = {"checkpoint": checkpoint_path, "period_losses": period_losses, **metrics}

    out_path = Path(checkpoint_path).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Eval results written → {out_path}")
    return results
