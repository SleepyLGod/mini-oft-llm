#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mini_oft_llm.data import prepare_firefly_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Firefly dataset into conversational JSONL splits.")
    parser.add_argument("--output-dir", type=str, default="data/firefly_prepared")
    parser.add_argument("--dataset-name", type=str, default="YeungNLP/firefly-train-1.1M")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_firefly_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
