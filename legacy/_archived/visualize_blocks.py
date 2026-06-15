"""Phase 1 — INCA block-chain diagram generator.

Loads a saved .pt checkpoint and renders an annotated architecture
diagram showing the frozen / trainable block chain, selector, decoder,
and per-block parameter counts.

Usage
-----
    python -m Phase1.scripts.visualize_inca \
        --checkpoint Phase1/results/<run>/inca_v2_final.pt \
        [--out diagram.png]          # default: same folder as checkpoint

The script is intentionally self-contained (only stdlib + torch +
matplotlib) so it can be run without the rest of the Phase1 package.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch


# ── colour palette ────────────────────────────────────────────────────────────

C_BG          = "#F7F9FC"
C_FROZEN      = "#B0C4DE"   # light steel-blue  — frozen block
C_FROZEN_TXT  = "#1A3A5C"
C_TRAINABLE   = "#F4A261"   # warm orange        — trainable block
C_TRAINABLE_TXT = "#7B2D00"
C_EMPTY       = "#E8E8E8"   # light grey         — unused slot
C_EMPTY_TXT   = "#888888"
C_EMBED       = "#A8D5A2"   # mint green         — embedding
C_SELECTOR    = "#C9B1D9"   # soft purple        — selector
C_NORM        = "#A2C4D9"   # sky blue           — layer norm
C_DECODER     = "#F7C6A3"   # peach              — decoder
C_LMHEAD      = "#FDE8A8"   # pale yellow        — LM head
C_ARROW       = "#555555"
C_TITLE       = "#1A1A2E"
C_STAT_BG     = "#EDF2FB"
C_BORDER      = "#CCCCCC"


# ── helpers ───────────────────────────────────────────────────────────────────

def _count_params(state_dict: Dict[str, Any]) -> int:
    return sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _rounded_box(ax, x, y, w, h, fc, ec="#888888", lw=1.4, radius=0.03,
                 zorder=3):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, zorder=2,
           arrowstyle="-|>", mutation_scale=12):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=lw,
            mutation_scale=mutation_scale,
        ),
        zorder=zorder,
    )


def _label(ax, x, y, text, fontsize=9, color="#1A1A2E", ha="center",
           va="center", weight="normal", zorder=5):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
            fontweight=weight, zorder=zorder)


# ── main diagram builder ──────────────────────────────────────────────────────

def build_diagram(ckpt_path: str, out_path: Optional[str] = None) -> str:
    """Load checkpoint and render diagram.  Returns path of saved PNG."""

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg            = ckpt.get("cfg", {})
    manager_state  = ckpt.get("manager_state", {})
    base_state     = ckpt.get("base_model_state", {})
    global_step    = ckpt.get("global_opt_step", "?")

    # ── derive architecture facts ─────────────────────────────────────
    n_blocks_used  = manager_state.get("n_blocks", 1)
    blocks_sd      = manager_state.get("blocks_state", [])
    selector_sd    = manager_state.get("selector_state", {})
    n_max_blocks   = cfg.get("n_max_blocks", "?")
    layers_per_blk = cfg.get("layers_per_block", "?")
    model_name     = cfg.get("model_name", "unknown")
    dataset        = cfg.get("dataset", "unknown")
    seed           = cfg.get("seed", "?")

    # param counts
    block_params   = [_count_params(sd) for sd in blocks_sd]
    selector_params = _count_params(selector_sd)

    # split base model into embedding / decoder / lm_head buckets
    embed_params  = sum(v.numel() for k, v in base_state.items()
                        if "shared" in k or "embed_tokens" in k)
    decoder_params = sum(v.numel() for k, v in base_state.items()
                         if k.startswith("decoder"))
    lmhead_params  = sum(v.numel() for k, v in base_state.items()
                         if "lm_head" in k)
    total_params   = sum(v.numel() for v in base_state.values()
                         if hasattr(v, "numel"))
    # trainable = last block + selector
    trainable_params = block_params[-1] + selector_params if block_params else 0

    n_max_int = n_max_blocks if isinstance(n_max_blocks, int) else 4
    n_empty   = max(0, n_max_int - n_blocks_used)

    # ── figure layout ──────────────────────────────────────────────────
    total_cols = n_blocks_used + n_empty
    fig_w = max(12, total_cols * 2.6 + 4)
    fig_h = 11
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # ── title banner ───────────────────────────────────────────────────
    banner_h = 0.7
    _rounded_box(ax, 0.3, fig_h - 1.0, fig_w - 0.6, banner_h,
                 fc="#1A1A2E", ec="#1A1A2E", lw=0, radius=0.15, zorder=2)
    _label(ax, fig_w / 2, fig_h - 0.65,
           f"INCA v2  ·  {model_name}  ·  dataset: {dataset}  ·  "
           f"opt_steps: {global_step}  ·  seed: {seed}",
           fontsize=11, color="white", weight="bold")

    # ── vertical layout constants ──────────────────────────────────────
    y_input    = fig_h - 1.85
    y_embed    = fig_h - 3.05
    y_blocks   = fig_h - 5.25
    y_selector = fig_h - 7.00
    y_norm     = fig_h - 8.10
    y_decoder  = fig_h - 9.20
    y_lmhead   = fig_h - 10.10

    box_h   = 1.05
    box_w   = 2.00
    # x centres for each block column
    left_margin = 1.5
    col_step    = (fig_w - left_margin - 1.5) / max(total_cols, 1)
    block_cx    = [left_margin + (i + 0.5) * col_step for i in range(total_cols)]
    chain_cx    = (block_cx[0] + block_cx[n_blocks_used - 1]) / 2  # centre of used blocks
    mid_x       = fig_w / 2

    # ── "Input Tokens" label ───────────────────────────────────────────
    _label(ax, mid_x, y_input + 0.15, "Input Tokens",
           fontsize=10, color=C_TITLE, weight="bold")
    _arrow(ax, mid_x, y_input - 0.05, mid_x, y_embed + box_h + 0.1)

    # ── Embedding box ─────────────────────────────────────────────────
    ex = mid_x - box_w / 2
    _rounded_box(ax, ex, y_embed, box_w, box_h, fc=C_EMBED, ec="#4A7C59", lw=1.5)
    _label(ax, mid_x, y_embed + 0.62, "Embedding", fontsize=10,
           color="#1E4D2B", weight="bold")
    _label(ax, mid_x, y_embed + 0.30, "shared · frozen", fontsize=8,
           color="#3A6B45")
    _label(ax, mid_x, y_embed + 0.06, _fmt_params(embed_params) + " params",
           fontsize=8, color="#3A6B45")

    # ── Architecture (a): sequential chain + original-embedding skip ──
    #
    #  original embeddings
    #       |
    #  +----+--------+--------+------+   ← skip bus (dashed)
    #  |    |        |        |
    #  v    v        v        v
    #  [B0] --proj-> [B1] --proj-> [Bc]
    #
    y_skip_bus = y_embed - 0.35        # dashed skip bus below embedding
    bus_x0 = block_cx[0]
    bus_x1 = block_cx[n_blocks_used - 1]

    # Embedding → skip bus stem
    ax.plot([mid_x, mid_x], [y_embed - 0.05, y_skip_bus],
            color="#4A7C59", lw=1.6, zorder=2)
    # Dashed horizontal skip bus across all used blocks
    ax.plot([bus_x0, bus_x1], [y_skip_bus, y_skip_bus],
            color="#4A7C59", lw=1.8, linestyle="--", dashes=(5, 3), zorder=2)
    _label(ax, (bus_x0 + bus_x1) / 2, y_skip_bus + 0.16,
           "original-embedding skip  (persistent residual to every block)",
           fontsize=7.5, color="#2E6B3E")

    # Dashed drop arrows: skip bus → top of each block
    for cx in block_cx[:n_blocks_used]:
        ax.annotate("", xy=(cx, y_blocks + box_h + 0.02),
                    xytext=(cx, y_skip_bus),
                    arrowprops=dict(arrowstyle="-|>", color="#4A7C59",
                                   lw=1.2, mutation_scale=10,
                                   linestyle="dashed"),
                    zorder=2)

    # Sequential chain: Block 0 → proj → Block 1 → proj → ...
    # Arrow from Block i right edge → proj label → Block i+1 left edge
    y_chain = y_blocks + box_h * 0.45   # mid-height of block boxes
    for i in range(n_blocks_used - 1):
        cx_src  = block_cx[i]   + box_w / 2
        cx_dst  = block_cx[i+1] - box_w / 2
        cx_mid  = (cx_src + cx_dst) / 2
        # Chain arrow
        _arrow(ax, cx_src, y_chain, cx_dst, y_chain,
               color="#333333", lw=1.8, arrowstyle="-|>", mutation_scale=13)
        # proj label
        _rounded_box(ax, cx_mid - 0.28, y_chain - 0.17, 0.56, 0.34,
                     fc="#F0E6FF", ec="#7B2FBE", lw=1.0, radius=0.05, zorder=4)
        _label(ax, cx_mid, y_chain - 0.00, f"proj{i+1}",
               fontsize=7.5, color="#4B0082", weight="bold")

    # Direct drop arrow: embedding → Block 0 (no proj for first block)
    _arrow(ax, block_cx[0], y_skip_bus,
           block_cx[0], y_blocks + box_h + 0.02,
           color="#4A7C59", lw=1.5)

    # ── Block boxes ───────────────────────────────────────────────────
    for i in range(total_cols):
        cx   = block_cx[i]
        bx   = cx - box_w / 2
        is_used      = i < n_blocks_used
        is_trainable = (i == n_blocks_used - 1) and is_used

        if not is_used:
            _rounded_box(ax, bx, y_blocks, box_w, box_h,
                         fc=C_EMPTY, ec=C_BORDER, lw=1.0, zorder=3)
            _label(ax, cx, y_blocks + 0.55, f"Slot {i}", fontsize=9,
                   color=C_EMPTY_TXT)
            _label(ax, cx, y_blocks + 0.30, "(unused)", fontsize=8,
                   color=C_EMPTY_TXT)
        elif is_trainable:
            _rounded_box(ax, bx, y_blocks, box_w, box_h,
                         fc=C_TRAINABLE, ec="#B85C00", lw=2.0, zorder=3)
            _label(ax, cx, y_blocks + 0.80, f"Block {i}  ★ TRAINABLE",
                   fontsize=9, color=C_TRAINABLE_TXT, weight="bold")
            _label(ax, cx, y_blocks + 0.52, f"{layers_per_blk} layers",
                   fontsize=8.5, color=C_TRAINABLE_TXT)
            p = block_params[i] if i < len(block_params) else 0
            _label(ax, cx, y_blocks + 0.27, _fmt_params(p) + " params",
                   fontsize=8, color=C_TRAINABLE_TXT)
            _label(ax, cx, y_blocks + 0.06, "gradients ON",
                   fontsize=7.5, color="#8B3A00")
        else:
            _rounded_box(ax, bx, y_blocks, box_w, box_h,
                         fc=C_FROZEN, ec="#2B4F80", lw=1.5, zorder=3)
            _label(ax, cx, y_blocks + 0.80, f"Block {i}  ❄ FROZEN",
                   fontsize=9, color=C_FROZEN_TXT, weight="bold")
            _label(ax, cx, y_blocks + 0.52, f"{layers_per_blk} layers",
                   fontsize=8.5, color=C_FROZEN_TXT)
            p = block_params[i] if i < len(block_params) else 0
            _label(ax, cx, y_blocks + 0.27, _fmt_params(p) + " params",
                   fontsize=8, color=C_FROZEN_TXT)
            _label(ax, cx, y_blocks + 0.06, "no grad  (torch.no_grad)",
                   fontsize=7.5, color="#2B4F80")

    # ── Fan-in bus: each block output → Selector ──────────────────────
    sel_cx = chain_cx
    sel_w  = max(2.2, n_blocks_used * col_step * 0.85)
    sel_x  = sel_cx - sel_w / 2
    y_bus_in = y_blocks - 0.50

    for cx in block_cx[:n_blocks_used]:
        ax.plot([cx, cx], [y_blocks - 0.02, y_bus_in],
                color=C_ARROW, lw=1.5, zorder=2)
    ax.plot([bus_x0, bus_x1], [y_bus_in, y_bus_in],
            color=C_ARROW, lw=2.2, zorder=2)
    _label(ax, (bus_x0 + bus_x1) / 2, y_bus_in - 0.18,
           "all block outputs aggregated  (CrossAttentionSelector)",
           fontsize=7.5, color="#555555")
    _arrow(ax, sel_cx, y_bus_in,
           sel_cx, y_selector + box_h + 0.05,
           color=C_ARROW, lw=1.8)

    # ── Selector box ─────────────────────────────────────────────────
    _rounded_box(ax, sel_x, y_selector, sel_w, box_h,
                 fc=C_SELECTOR, ec="#6A3D9A", lw=1.8, zorder=3)
    _label(ax, sel_cx, y_selector + 0.72, "CrossAttention Selector",
           fontsize=9.5, color="#3B006E", weight="bold")
    _label(ax, sel_cx, y_selector + 0.44,
           f"softmax-gated aggregation  ·  {n_blocks_used} block(s)",
           fontsize=8, color="#5A007F")
    _label(ax, sel_cx, y_selector + 0.18,
           _fmt_params(selector_params) + " params  ·  gradients ON",
           fontsize=8, color="#5A007F")

    # selector → layer norm
    _arrow(ax, sel_cx, y_selector - 0.05, mid_x, y_norm + box_h + 0.08)

    # ── LayerNorm box ────────────────────────────────────────────────
    _rounded_box(ax, mid_x - box_w / 2, y_norm, box_w, box_h * 0.70,
                 fc=C_NORM, ec="#1C6998", lw=1.5)
    _label(ax, mid_x, y_norm + 0.40, "Final LayerNorm",
           fontsize=9, color="#0A3D62", weight="bold")
    _label(ax, mid_x, y_norm + 0.16, "encoder output normalisation",
           fontsize=7.5, color="#1C6998")

    # layer norm → decoder
    _arrow(ax, mid_x, y_norm - 0.05, mid_x, y_decoder + box_h * 0.75 + 0.08)

    # ── Decoder box ──────────────────────────────────────────────────
    _rounded_box(ax, mid_x - box_w / 2, y_decoder, box_w, box_h * 0.75,
                 fc=C_DECODER, ec="#A0522D", lw=1.5)
    _label(ax, mid_x, y_decoder + 0.45, "T5 Decoder",
           fontsize=9, color="#6B2A00", weight="bold")
    _label(ax, mid_x, y_decoder + 0.20,
           _fmt_params(decoder_params) + " params  ·  frozen",
           fontsize=7.5, color="#8B4513")

    # decoder → lm head
    _arrow(ax, mid_x, y_decoder - 0.05, mid_x, y_lmhead + box_h * 0.60 + 0.08)

    # ── LM Head box ──────────────────────────────────────────────────
    _rounded_box(ax, mid_x - box_w / 2, y_lmhead, box_w, box_h * 0.60,
                 fc=C_LMHEAD, ec="#B8860B", lw=1.5)
    _label(ax, mid_x, y_lmhead + 0.35, "LM Head  (linear)",
           fontsize=9, color="#6B4F00", weight="bold")
    _label(ax, mid_x, y_lmhead + 0.14,
           _fmt_params(lmhead_params) + " params  ·  frozen",
           fontsize=7.5, color="#8B6914")

    # lm head → output
    _arrow(ax, mid_x, y_lmhead - 0.05, mid_x, 0.35)
    _label(ax, mid_x, 0.22, "Output logits", fontsize=9.5,
           color=C_TITLE, weight="bold")

    # ── Architecture note ─────────────────────────────────────────────
    # Placed to the left of the block chain if space permits
    note_x = 0.25
    note_y = y_blocks - 0.10
    note_w = max(0.1, block_cx[0] - box_w / 2 - 0.5)
    if note_w > 1.0:
        note_h = 1.9
        _rounded_box(ax, note_x, note_y, note_w, note_h,
                     fc="#FFFBEA", ec="#D4A017", lw=1.1, radius=0.08, zorder=1)
        _label(ax, note_x + note_w / 2, note_y + note_h - 0.26,
               "Architecture (a)", fontsize=8, color="#6B4F00", weight="bold")
        for dy, txt in [
            (0.58, "Sequential chain"),
            (0.42, "+ embedding skip."),
            (0.26, "proj_i transfers"),
            (0.10, "frozen knowledge."),
        ]:
            _label(ax, note_x + note_w / 2, note_y + dy, txt,
                   fontsize=7, color="#6B4F00")

    # ── Stats panel (right side) ──────────────────────────────────────
    sp_x  = fig_w - 3.4
    sp_y  = fig_h - 7.5
    sp_w  = 3.0
    sp_h  = 5.5
    _rounded_box(ax, sp_x, sp_y, sp_w, sp_h,
                 fc=C_STAT_BG, ec=C_BORDER, lw=1.2, radius=0.12, zorder=1)

    lines = [
        ("Run statistics", True),
        ("", False),
        (f"Model:  {model_name.split('/')[-1]}", False),
        (f"Dataset: {dataset}", False),
        (f"Seed:    {seed}", False),
        (f"Opt steps: {global_step}", False),
        ("", False),
        ("Architecture", True),
        ("", False),
        (f"Blocks used:   {n_blocks_used} / {n_max_blocks}", False),
        (f"Layers/block:  {layers_per_blk}", False),
        (f"Selector:      {_fmt_params(selector_params)}", False),
        ("", False),
        ("Parameters", True),
        ("", False),
        (f"Total:       {_fmt_params(total_params)}", False),
        (f"Trainable:   {_fmt_params(trainable_params)}", False),
        (f"  (last block + selector)", False),
        (f"Frozen:      {_fmt_params(total_params - trainable_params)}", False),
    ]
    ty = sp_y + sp_h - 0.35
    for text, bold in lines:
        if not text:
            ty -= 0.12
            continue
        _label(ax, sp_x + sp_w / 2, ty, text,
               fontsize=8.0, color="#1A1A2E",
               weight="bold" if bold else "normal")
        ty -= 0.24

    # ── Legend ────────────────────────────────────────────────────────
    lx, ly = 0.35, 0.82
    for fc, ec, label in [
        (C_FROZEN,    "#2B4F80", "Frozen block"),
        (C_TRAINABLE, "#B85C00", "Trainable block"),
        (C_EMPTY,     C_BORDER,  "Unused slot"),
        (C_SELECTOR,  "#6A3D9A", "Selector"),
        (C_EMBED,     "#4A7C59", "Embedding"),
    ]:
        _rounded_box(ax, lx, ly - 0.04, 0.22, 0.22,
                     fc=fc, ec=ec, lw=1.0, radius=0.03, zorder=4)
        _label(ax, lx + 0.36, ly + 0.07, label,
               fontsize=7.5, ha="left", color="#1A1A2E")
        lx += 2.05

    # ── save ──────────────────────────────────────────────────────────
    if out_path is None:
        out_path = str(Path(ckpt_path).parent / "block_diagram.png")

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close(fig)
    print(f"Diagram saved → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate INCA block diagram from checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to inca_v2_final.pt")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: block_diagram.png beside checkpoint)")
    args = p.parse_args()
    build_diagram(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
