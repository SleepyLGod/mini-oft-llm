from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool,
    use_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bfloat16",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        dtype = getattr(torch, bnb_4bit_compute_dtype)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        quantization_config=quant_config,
        device_map="auto" if quant_config else None,
    )

    if quant_config is None:
        device = _resolve_device()
        model.to(device)

    model.eval()
    return tokenizer, model


def generate_for_prompt(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt")

    # device_map='auto' keeps model sharded; fallback uses model.device.
    target_device = getattr(model, "device", None)
    if target_device is not None:
        encoded = {k: v.to(target_device) for k, v in encoded.items()}

    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = out[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_before_after_generation(
    base_model_name_or_path: str,
    adapter_path: str,
    prompt_rows: list[dict[str, Any]],
    output_jsonl: str | Path,
    trust_remote_code: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    use_4bit: bool = False,
) -> list[dict[str, Any]]:
    tokenizer, base_model = load_model_and_tokenizer(
        base_model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_4bit=use_4bit,
    )

    results: list[dict[str, Any]] = []
    for row in prompt_rows:
        prompt = row["prompt"]
        before = generate_for_prompt(
            tokenizer,
            base_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(
            {
                "id": row.get("id"),
                "group": row.get("group", "unknown"),
                "prompt": prompt,
                "base_output": before,
                "oft_output": "",
            }
        )

    tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    tuned_model.eval()

    for row in prompt_rows:
        prompt = row["prompt"]
        after = generate_for_prompt(
            tokenizer,
            tuned_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for result in results:
            if result["id"] == row.get("id"):
                result["oft_output"] = after
                break

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def compute_token_level_nll(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        target_device = getattr(model, "device", None)
        if target_device is not None:
            enc = {k: v.to(target_device) for k, v in enc.items()}

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        with torch.inference_mode():
            outputs = model(**enc, labels=labels)
            loss = float(outputs.loss.detach().cpu().item())

        # HuggingFace internally shifts labels by 1 for causal LM:
        #   shift_labels = labels[:, 1:]
        # so the loss denominator is tokens where labels[:, 1:] != -100.
        # We must match that count to correctly reconstruct total loss.
        shifted_labels = labels[:, 1:]
        valid_tokens = int((shifted_labels != -100).sum().detach().cpu().item())
        total_loss += loss * valid_tokens
        total_tokens += valid_tokens

    mean_nll = total_loss / max(total_tokens, 1)
    ppl = float(math.exp(mean_nll)) if mean_nll < 20 else float("inf")
    return {
        "num_texts": len(texts),
        "num_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": ppl,
    }
