#!/usr/bin/env python3
"""Plot training / evaluation loss curves.

Supports two modes:
  1. Single run:
       python plot_training_curves.py --trainer-state STATE --output-png OUT
  2. Comparison (multiple runs overlaid on one figure):
       python plot_training_curves.py --compare \
         --trainer-state outputs/h100_main/trainer_state.json \
         --label "block_size=32 (main)" \
         --trainer-state outputs/h100_ablation_block16/trainer_state.json \
         --label "block_size=16 (ablation)" \
         --output-png artifacts/loss_comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_curves(state_path: Path):
    """Return (train_steps, train_loss, eval_steps, eval_loss) from a trainer_state.json."""
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row:
            train_steps.append(step)
            train_loss.append(row["loss"])
        if "eval_loss" in row:
            eval_steps.append(step)
            eval_loss.append(row["eval_loss"])

    return train_steps, train_loss, eval_steps, eval_loss


COLORS = [
    ("#1f77b4", "#aec7e8"),  # blue pair
    ("#ff7f0e", "#ffbb78"),  # orange pair
    ("#2ca02c", "#98df8a"),  # green pair
    ("#d62728", "#ff9896"),  # red pair
]


# ---------------------------------------------------------------------------
# Single-run plot
# ---------------------------------------------------------------------------

def plot_single(state_path: Path, out: Path) -> None:
    train_steps, train_loss, eval_steps, eval_loss = _extract_curves(state_path)

    plt.figure(figsize=(9, 5))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training / Evaluation Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_compare(state_paths: list[Path], labels: list[str], out: Path) -> None:
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (sp, lbl) in enumerate(zip(state_paths, labels)):
        train_steps, train_loss, eval_steps, eval_loss = _extract_curves(sp)
        c_dark, c_light = COLORS[idx % len(COLORS)]

        if train_steps:
            ax_train.plot(train_steps, train_loss, color=c_dark, label=lbl, linewidth=1.4)
        if eval_steps:
            ax_eval.plot(eval_steps, eval_loss, color=c_dark, marker="o", markersize=3, label=lbl, linewidth=1.4)

    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Loss")
    ax_train.set_title("Train Loss Comparison")
    ax_train.legend()
    ax_train.grid(alpha=0.3)

    ax_eval.set_xlabel("Step")
    ax_eval.set_ylabel("Loss")
    ax_eval.set_title("Eval Loss Comparison")
    ax_eval.legend()
    ax_eval.grid(alpha=0.3)

    fig.suptitle("OFT Ablation: Main vs Ablation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/eval loss curves from trainer_state.json")
    parser.add_argument("--trainer-state", type=str, action="append", required=True,
                        help="Path(s) to trainer_state.json. Repeat for comparison mode.")
    parser.add_argument("--label", type=str, action="append", default=None,
                        help="Label per trainer-state (used in comparison mode).")
    parser.add_argument("--compare", action="store_true",
                        help="Enable comparison mode (overlay multiple runs).")
    parser.add_argument("--output-png", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.compare or len(args.trainer_state) > 1:
        state_paths = [Path(p) for p in args.trainer_state]
        labels = args.label or [Path(p).parent.name for p in args.trainer_state]
        # Pad labels if fewer than state_paths
        while len(labels) < len(state_paths):
            labels.append(Path(args.trainer_state[len(labels)]).parent.name)
        plot_compare(state_paths, labels, out)
    else:
        plot_single(Path(args.trainer_state[0]), out)


if __name__ == "__main__":
    main()
