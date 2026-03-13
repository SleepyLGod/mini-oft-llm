from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import OFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from mini_oft_llm.config import ProjectConfig


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def build_quantization_config(cfg: ProjectConfig) -> BitsAndBytesConfig | None:
    if not cfg.model.use_4bit:
        return None

    compute_dtype = resolve_dtype(cfg.model.bnb_4bit_compute_dtype)
    if compute_dtype == "auto":
        compute_dtype = torch.bfloat16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_tokenizer_and_model(cfg: ProjectConfig):
    quantization_config = build_quantization_config(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=resolve_dtype(cfg.model.dtype),
        device_map="auto" if quantization_config else None,
    )
    return tokenizer, model


def load_local_sft_datasets(cfg: ProjectConfig):
    data_root = Path(cfg.data.prepared_dir)
    data_files = {
        "train": str(data_root / cfg.data.train_file),
        "validation": str(data_root / cfg.data.val_file),
        "test": str(data_root / cfg.data.test_file),
    }
    ds = load_dataset("json", data_files=data_files)

    # Keep only the "messages" column so TRL recognises the dataset as
    # conversational (required for assistant_only_loss=True).
    keep_cols = {"messages"}
    for split in ds:
        extra = set(ds[split].column_names) - keep_cols
        if extra:
            ds[split] = ds[split].remove_columns(list(extra))

    return ds["train"], ds["validation"], ds["test"]


def build_oft_config(cfg: ProjectConfig) -> OFTConfig:
    return OFTConfig(
        task_type="CAUSAL_LM",
        target_modules=cfg.oft.target_modules,
        oft_block_size=cfg.oft.oft_block_size,
        use_cayley_neumann=cfg.oft.use_cayley_neumann,
        module_dropout=cfg.oft.module_dropout,
        bias=cfg.oft.bias,
        coft=cfg.oft.coft,
        eps=cfg.oft.eps,
    )


def build_sft_args(cfg: ProjectConfig) -> SFTConfig:
    t = cfg.training
    kwargs: dict[str, Any] = {
        "output_dir": t.output_dir,
        "run_name": cfg.run_name,
        "num_train_epochs": t.num_train_epochs,
        "learning_rate": t.learning_rate,
        "warmup_ratio": t.warmup_ratio,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.per_device_eval_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "eval_strategy": t.eval_strategy,
        "eval_steps": t.eval_steps,
        "save_steps": t.save_steps,
        "logging_steps": t.logging_steps,
        "save_total_limit": t.save_total_limit,
        "max_steps": t.max_steps,
        "seed": t.seed,
        "lr_scheduler_type": t.lr_scheduler_type,
        "weight_decay": t.weight_decay,
        "max_grad_norm": t.max_grad_norm,
        "gradient_checkpointing": t.gradient_checkpointing,
        "bf16": t.bf16,
        "fp16": t.fp16,
        "report_to": t.report_to,
        "packing": t.packing,
        "max_length": cfg.data.max_length,
    }

    # Keep compatibility with older/newer TRL versions.
    init_sig = inspect.signature(SFTConfig.__init__)
    params = init_sig.parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "max_length" not in params:
        kwargs.pop("max_length", None)
    if "packing" not in params:
        kwargs.pop("packing", None)
    if "assistant_only_loss" in init_sig.parameters:
        kwargs["assistant_only_loss"] = t.assistant_only_loss

    return SFTConfig(**kwargs)


def build_sft_trainer(cfg: ProjectConfig) -> tuple[SFTTrainer, Any, Any, Any]:
    tokenizer, model = build_tokenizer_and_model(cfg)
    train_ds, val_ds, test_ds = load_local_sft_datasets(cfg)
    peft_config = build_oft_config(cfg)
    sft_args = build_sft_args(cfg)

    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "peft_config": peft_config,
    }

    # TRL versions use either `processing_class` or `tokenizer`.
    init_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in init_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    return trainer, tokenizer, val_ds, test_ds
