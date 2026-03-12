#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/eval loss curves from trainer_state.json")
    parser.add_argument("--trainer-state", type=str, required=True)
    parser.add_argument("--output-png", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = Path(args.trainer_state)
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

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

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
