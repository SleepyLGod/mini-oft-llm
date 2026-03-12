#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from mini_oft_llm.eval import load_prompts, run_before_after_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate before/after outputs for a fixed prompt set.")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, default="prompts/eval_prompts_zh.jsonl")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompt_file)
    rows = run_before_after_generation(
        base_model_name_or_path=args.base_model,
        adapter_path=args.adapter_path,
        prompt_rows=prompts,
        output_jsonl=args.output_jsonl,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_4bit=args.use_4bit,
    )

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Before vs After (OFT)\n\n")
        for row in rows:
            f.write(f"## {row['id']} - {row['group']}\n\n")
            f.write("**Prompt**\n\n")
            f.write(row["prompt"] + "\n\n")
            f.write("**Base Model Output**\n\n")
            f.write(row["base_output"] + "\n\n")
            f.write("**OFT Output**\n\n")
            f.write(row["oft_output"] + "\n\n")

    print(json.dumps({"num_prompts": len(rows), "output_jsonl": args.output_jsonl, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
