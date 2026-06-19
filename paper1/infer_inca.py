"""INCA checkpoint inference  —  scripts/infer_inca.py

Load any INCA .pt checkpoint produced by training/inca_trainer.py and run
interactive / batch generation.  Optionally shows which block the selector
routes each token to (useful for sanity-checking that the grown blocks are
being used and not just ignored).

Checkpoint format (written by inca_trainer.py)
----------------------------------------------
  {
    "period":           str,   # last completed period id
    "block_idx":        int,   # how many blocks exist
    "global_opt_step":  int,
    "manager_state":    dict,  # INCALayerManager.manager_state()
    "base_model_state": dict,  # T5ForConditionalGeneration.state_dict()
    "optimizer_state":  dict,
    "cfg":              dict,  # dataclasses.asdict(INCAConfig)
  }

Usage examples
--------------
  # Interactive REPL (type prompts, empty line to quit)
  python scripts/infer_inca.py results/paper_b/inca_v2_*/inca_v2_final.pt

  # Single prompt from CLI
  python scripts/infer_inca.py results/paper_b/inca_v2_*/inca_period_P1_metamath.pt \\
      --prompt "solve: A car travels 150 miles in 3 hours. What is its average speed?"

  # Batch mode — one raw prompt per line in a text file
  python scripts/infer_inca.py checkpoint.pt \\
      --prompts-file my_prompts.txt --output results.jsonl

  # Show block-routing weights (which block each sample routes to)
  python scripts/infer_inca.py checkpoint.pt --routing \\
      --prompt "code: Write a Python function to reverse a linked list."

  # Compare with un-fine-tuned base FLAN-T5 side by side
  python scripts/infer_inca.py checkpoint.pt --compare-base \\
      --prompt "solve: What is the 10th Fibonacci number?"

  # Auto-prefix: skips manual "solve:" / "code:" prefix — detected from period name
  python scripts/infer_inca.py checkpoint.pt --auto-prefix \\
      --prompt "A train leaves Chicago at 9am …"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# ── repo root on path ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from models.inca.config       import INCAConfig
from models.inca.layer_manager import INCALayerManager


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device: str) -> Tuple[
    T5ForConditionalGeneration,
    INCALayerManager,
    AutoTokenizer,
    INCAConfig,
    dict,
]:
    """Load an INCA .pt checkpoint and return (model, manager, tokenizer, cfg, meta).

    Works with both period checkpoints (inca_period_*.pt) and the final
    checkpoint (inca_v2_final.pt).
    """
    print(f"[infer] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Reconstruct config ──────────────────────────────────────────────
    cfg_dict = ckpt["cfg"]
    # dataclasses.asdict produces a plain dict; rebuild INCAConfig from it.
    # Use only fields that INCAConfig knows about (ignore unknown keys).
    _known = {f.name for f in dataclasses.fields(INCAConfig)}
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in _known})

    print(f"[infer]   model_name   : {cfg.model_name}")
    print(f"[infer]   period saved : {ckpt.get('period', 'unknown')}")
    print(f"[infer]   block_idx    : {ckpt.get('block_idx', 0)}")
    print(f"[infer]   selector     : {cfg.selector}")

    # ── Tokenizer + base model ──────────────────────────────────────────
    tokenizer  = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)
    base_model.to(device)

    # ── INCA manager (grows to match saved block count) ─────────────────
    manager = INCALayerManager(base_model, cfg).to(device)

    # ── Restore weights ─────────────────────────────────────────────────
    base_model.load_state_dict(ckpt["base_model_state"])
    manager.load_manager_state(ckpt["manager_state"])

    base_model.eval()
    manager.eval()

    n_blocks = manager.n_blocks
    total_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"[infer]   blocks       : {n_blocks}")
    print(f"[infer]   total params : {total_params:.1f}M")
    print(f"[infer]   device       : {device}")
    print()

    meta = {
        "period":         ckpt.get("period", "unknown"),
        "block_idx":      ckpt.get("block_idx", 0),
        "global_opt_step": ckpt.get("global_opt_step", 0),
        "n_blocks":       n_blocks,
    }
    return base_model, manager, tokenizer, cfg, meta


# ──────────────────────────────────────────────────────────────────────────────
# Auto-prefix detection
# ──────────────────────────────────────────────────────────────────────────────

# Map period name → input prefix
_PERIOD_PREFIX: Dict[str, str] = {
    "P1_metamath":   "solve: ",
    "P1_math":       "solve: ",
    "P1_gsm8k":      "solve: ",
    "P2_evol_code":  "code: ",
    "P2_code":       "code: ",
    "P3_science":    "answer: ",
    "P4_medical":    "answer: ",
    "P5_commonsense": "answer: ",
    "P1_trivia":     "answer: ",
}


def _auto_prefix(text: str, period: str) -> str:
    """Prepend the domain prefix if the text doesn't already have one."""
    prefix = _PERIOD_PREFIX.get(period, "")
    for known in _PERIOD_PREFIX.values():
        if text.startswith(known):
            return text   # already has a prefix
    return prefix + text


# ──────────────────────────────────────────────────────────────────────────────
# Core generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_one(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    tokenizer,
    prompt: str,
    device: str,
    max_input_length: int = 256,
    max_new_tokens:   int = 256,
) -> str:
    """Run greedy decoding for a single prompt string.  Returns generated text."""
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    ).to(device)

    enc_hidden = manager(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )
    enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
    gen_ids = model.generate(
        encoder_outputs=enc_out,
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────────────
# Block routing weights  (EmbeddingQuerySelector internals)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_routing_weights(
    manager: INCALayerManager,
    tokenizer,
    prompt: str,
    device: str,
    max_input_length: int = 256,
) -> Optional[List[float]]:
    """Return per-block routing weights [w_0, w_1, …] for *prompt*.

    Only implemented for EmbeddingQuerySelector (the default).
    Returns None if a different selector is used or there is only 1 block.

    Routing weight w_i = how much block i's output contributes to the final
    encoder representation passed to the decoder.  Computed by replicating
    the EmbeddingQuerySelector's softmax-over-blocks logic.
    """
    if manager.n_blocks == 1:
        return [1.0]

    sel = manager.selector
    if not hasattr(sel, "k_proj"):
        # CrossAttentionSelector or WeightedSumSelector — handle separately
        return _routing_fallback(manager, tokenizer, prompt, device, max_input_length)

    # ── Tokenise ────────────────────────────────────────────────────────
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    ).to(device)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # ── Rerun manager forward, capturing per-block outputs ─────────────
    with torch.no_grad():
        hidden          = manager.embed_tokens(input_ids)
        embedding_hidden = hidden.clone()
        chain_hidden    = hidden

        block_outputs: List[torch.Tensor] = []
        for i, block in enumerate(manager.blocks):
            if i > 0:
                proj = manager.inter_block_projs[i - 1]
                chain_hidden = proj(chain_hidden) + embedding_hidden
            h = block(chain_hidden, attention_mask=attention_mask)
            h = manager.final_layer_norm(h)
            block_outputs.append(h)
            chain_hidden = h

    # ── Replicate EmbeddingQuerySelector block-weight logic ────────────
    B, S, D = embedding_hidden.shape
    nh, dh   = sel.n_heads, sel.head_dim
    scale    = sel.scale

    Q = embedding_hidden.view(B, S, nh, dh).transpose(1, 2)  # (B, nh, S, dh)
    k_pad = attention_mask.unsqueeze(1).unsqueeze(2)           # (B, 1, 1, S)

    per_block_score: List[torch.Tensor] = []
    for out_i in block_outputs:
        K_i     = sel.k_proj(out_i).view(B, S, nh, dh).transpose(1, 2)
        scores_i = torch.matmul(Q, K_i.transpose(-2, -1)) * scale  # (B, nh, S, S)
        scores_i = scores_i.masked_fill(k_pad == 0, float("-inf"))
        # Mean diagonal Q·K alignment over real (non-padding) tokens
        diag_i  = scores_i.diagonal(dim1=-2, dim2=-1)               # (B, nh, S)
        am      = attention_mask.unsqueeze(1).float()                # (B, 1, S)
        denom   = am.sum(-1).clamp(min=1)                            # (B, 1)
        diag_i  = (diag_i * am).sum(-1) / denom                     # (B, nh)
        per_block_score.append(diag_i.mean(-1))                      # (B,)

    block_scores  = torch.stack(per_block_score, dim=-1)   # (B, n_blocks)
    block_weights = F.softmax(block_scores, dim=-1)         # (B, n_blocks)
    return block_weights[0].tolist()


def _routing_fallback(manager, tokenizer, prompt, device, max_input_length):
    """Routing weights for CrossAttention / WeightedSum selectors."""
    sel = manager.selector
    if hasattr(sel, "logits"):  # WeightedSumSelector
        n = manager.n_blocks
        logits  = sel.logits[:n] if len(sel.logits) >= n else F.pad(sel.logits, (0, n - len(sel.logits)))
        weights = F.softmax(logits, dim=0)
        return weights.tolist()
    # CrossAttentionSelector: would need full forward pass internals
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def _bar(weight: float, width: int = 20) -> str:
    filled = round(weight * width)
    return "█" * filled + "░" * (width - filled)


def _print_routing(weights: Optional[List[float]]) -> None:
    if weights is None:
        print("  [routing] (not available for this selector type)")
        return
    print(f"  [routing] {len(weights)} block(s)")
    for i, w in enumerate(weights):
        tag = " ← dominant" if w == max(weights) else ""
        print(f"    Block {i}  {_bar(w)}  {w:.3f}{tag}")


def _print_result(
    prompt: str,
    output: str,
    weights: Optional[List[float]] = None,
    base_output: Optional[str] = None,
    idx: Optional[int] = None,
) -> None:
    sep = "─" * 60
    header = f"[{idx}] " if idx is not None else ""
    print(f"\n{sep}")
    # Prompt — wrap long lines
    wrapped = textwrap.fill(prompt, width=70, subsequent_indent="         ")
    print(f"  {header}PROMPT : {wrapped}")
    print(f"  OUTPUT : {output}")
    if base_output is not None:
        print(f"  BASE   : {base_output}")
    if weights is not None:
        _print_routing(weights)
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Batch / file mode
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    tokenizer,
    cfg: INCAConfig,
    meta: dict,
    prompts: List[str],
    device: str,
    args: argparse.Namespace,
    base_model_frozen: Optional[T5ForConditionalGeneration] = None,
) -> List[dict]:
    """Run all prompts and return a list of result dicts."""
    results = []
    for i, raw_prompt in enumerate(prompts):
        if not raw_prompt.strip():
            continue

        prompt = _auto_prefix(raw_prompt.strip(), meta["period"]) if args.auto_prefix else raw_prompt.strip()

        output = generate_one(
            model, manager, tokenizer, prompt, device,
            max_input_length=cfg.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )

        weights = get_routing_weights(manager, tokenizer, prompt, device,
                                      cfg.max_input_length) if args.routing else None

        base_out = None
        if base_model_frozen is not None:
            # Baseline: run the frozen base T5 through a trivial identity manager
            base_enc  = tokenizer(prompt, truncation=True, max_length=cfg.max_input_length,
                                  return_tensors="pt").to(device)
            base_gen  = base_model_frozen.generate(
                input_ids=base_enc["input_ids"],
                attention_mask=base_enc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
            )
            base_out  = tokenizer.decode(base_gen[0], skip_special_tokens=True)

        _print_result(prompt, output, weights=weights, base_output=base_out, idx=i + 1)

        results.append({
            "idx":    i + 1,
            "prompt": prompt,
            "output": output,
            "base":   base_out,
            "routing_weights": weights,
            "period": meta["period"],
            "n_blocks": meta["n_blocks"],
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ──────────────────────────────────────────────────────────────────────────────

def run_repl(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    tokenizer,
    cfg: INCAConfig,
    meta: dict,
    device: str,
    args: argparse.Namespace,
    base_model_frozen: Optional[T5ForConditionalGeneration] = None,
) -> None:
    """Interactive prompt loop.  Empty line or Ctrl-C to quit."""
    period  = meta["period"]
    n_b     = meta["n_blocks"]
    prefix  = _PERIOD_PREFIX.get(period, "")
    print(f"\n{'='*60}")
    print(f"  INCA Inference REPL")
    print(f"  Period  : {period}")
    print(f"  Blocks  : {n_b}")
    print(f"  Selector: {cfg.selector}")
    print(f"  Device  : {device}")
    if prefix:
        print(f"  Auto-prefix: '{prefix}' (use --no-auto-prefix to disable)")
    print(f"  Type a prompt and press Enter.  Empty line or Ctrl-C to quit.")
    print(f"{'='*60}\n")

    history: List[str] = []
    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[infer] bye.")
            break
        if not raw:
            break

        prompt = _auto_prefix(raw, period) if args.auto_prefix else raw
        history.append(prompt)

        output  = generate_one(model, manager, tokenizer, prompt, device,
                               max_input_length=cfg.max_input_length,
                               max_new_tokens=args.max_new_tokens)
        weights = get_routing_weights(manager, tokenizer, prompt, device,
                                      cfg.max_input_length) if args.routing else None
        base_out = None
        if base_model_frozen is not None:
            benc = tokenizer(prompt, truncation=True, max_length=cfg.max_input_length,
                             return_tensors="pt").to(device)
            bgen = base_model_frozen.generate(
                input_ids=benc["input_ids"],
                attention_mask=benc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
            )
            base_out = tokenizer.decode(bgen[0], skip_special_tokens=True)

        _print_result(prompt, output, weights=weights, base_output=base_out,
                      idx=len(history))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference on an INCA .pt checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("checkpoint", help="Path to .pt checkpoint file")

    # Input modes (mutually exclusive)
    inp = p.add_mutually_exclusive_group()
    inp.add_argument("--prompt",       "-p", help="Single prompt string")
    inp.add_argument("--prompts-file", "-f", help="Text file — one prompt per line")

    # Generation settings
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Maximum tokens to generate (default: 256)")
    p.add_argument("--max-input-length", type=int, default=None,
                   help="Override max input length from checkpoint config")

    # Extras
    p.add_argument("--routing",       action="store_true",
                   help="Show per-block routing weights (EmbeddingQuerySelector)")
    p.add_argument("--compare-base",  action="store_true",
                   help="Also run the un-fine-tuned base FLAN-T5 for comparison")
    p.add_argument("--auto-prefix",   action="store_true", default=True,
                   help="Auto-prepend 'solve:'/'code:'/'answer:' based on period (default: on)")
    p.add_argument("--no-auto-prefix", dest="auto_prefix", action="store_false",
                   help="Disable auto-prefix — use prompt exactly as given")
    p.add_argument("--output", "-o",  help="Write results to a .jsonl file")
    p.add_argument("--device",        default=None,
                   help="Device: 'cuda', 'mps', 'cpu' (auto-detected if omitted)")
    return p.parse_args()


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args   = _parse_args()
    device = args.device or _auto_device()
    print(f"[infer] device = {device}")

    # ── Load INCA checkpoint ────────────────────────────────────────────
    model, manager, tokenizer, cfg, meta = load_checkpoint(
        args.checkpoint, device,
    )

    # Override max_input_length if requested
    if args.max_input_length:
        cfg.max_input_length = args.max_input_length

    # ── Optionally load frozen base model for comparison ───────────────
    base_model_frozen = None
    if args.compare_base:
        print("[infer] Loading un-fine-tuned base model for comparison …")
        base_model_frozen = T5ForConditionalGeneration.from_pretrained(cfg.model_name)
        base_model_frozen.to(device).eval()

    # ── Dispatch to mode ────────────────────────────────────────────────
    if args.prompt:
        results = run_batch(model, manager, tokenizer, cfg, meta,
                            prompts=[args.prompt], device=device, args=args,
                            base_model_frozen=base_model_frozen)

    elif args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as fh:
            lines = [l.rstrip("\n") for l in fh if l.strip()]
        print(f"[infer] Running {len(lines)} prompts from {args.prompts_file}")
        results = run_batch(model, manager, tokenizer, cfg, meta,
                            prompts=lines, device=device, args=args,
                            base_model_frozen=base_model_frozen)

    else:
        # Interactive REPL
        run_repl(model, manager, tokenizer, cfg, meta, device, args,
                 base_model_frozen=base_model_frozen)
        results = []

    # ── Save to JSONL if requested ──────────────────────────────────────
    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[infer] Results written to {out_path}  ({len(results)} entries)")


if __name__ == "__main__":
    main()
