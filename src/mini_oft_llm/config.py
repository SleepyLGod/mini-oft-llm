from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    prepared_dir: str
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"
    max_length: int = 1024


@dataclass
class ModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = True
    use_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    dtype: str = "auto"


@dataclass
class OFTRuntimeConfig:
    target_modules: str = "all-linear"
    oft_block_size: int = 32
    use_cayley_neumann: bool = True
    module_dropout: float = 0.0
    bias: str = "none"
    coft: bool = False
    eps: float = 6e-5


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 2
    max_steps: int = -1
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    packing: bool = False
    assistant_only_loss: bool = True
    report_to: str = "none"


@dataclass
class ProjectConfig:
    run_name: str
    data: DataConfig
    model: ModelConfig
    oft: OFTRuntimeConfig = field(default_factory=OFTRuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(payload: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def to_project_config(raw: dict[str, Any]) -> ProjectConfig:
    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    oft = OFTRuntimeConfig(**raw.get("oft", {}))
    training = TrainingConfig(**raw["training"])
    return ProjectConfig(run_name=raw["run_name"], data=data, model=model, oft=oft, training=training)
