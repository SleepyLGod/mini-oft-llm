#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mini_oft_llm.config import dump_yaml, load_yaml, to_project_config
from mini_oft_llm.training import build_sft_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OFT adapter using TRL SFTTrainer.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override-output-dir",
        type=str,
        default=None,
        help="Override training.output_dir in config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(args.config)

    if args.override_output_dir:
        raw_cfg["training"]["output_dir"] = args.override_output_dir

    cfg = to_project_config(raw_cfg)
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml(raw_cfg, output_dir / "resolved_config.yaml")

    trainer, _, _, _ = build_sft_trainer(cfg)

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))

    summary = {
        "output_dir": str(output_dir),
        "final_adapter_dir": str(final_adapter_dir),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
