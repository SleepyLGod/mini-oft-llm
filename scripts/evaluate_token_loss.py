#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mini_oft_llm.eval import compute_token_level_nll, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate token-level NLL / perplexity before and after OFT.")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True, help="Local JSONL with `messages` field.")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = load_dataset("json", data_files={"test": args.test_file})["test"]
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tokenizer, base_model = load_model_and_tokenizer(
        model_name_or_path=args.base_model,
        trust_remote_code=True,
        use_4bit=args.use_4bit,
    )

    texts: list[str] = []
    for item in ds:
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        texts.append(text)

    base_metrics = compute_token_level_nll(
        model=base_model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    oft_metrics = compute_token_level_nll(
        model=tuned_model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    delta_nll = oft_metrics["mean_nll"] - base_metrics["mean_nll"]
    summary = {
        "num_samples": len(texts),
        "base": base_metrics,
        "oft": oft_metrics,
        "delta_mean_nll": delta_nll,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
