from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def convert_firefly_record(record: dict[str, Any]) -> dict[str, Any] | None:
    user_input = _normalize_text(record.get("input"))
    assistant_output = _normalize_text(record.get("target"))
    kind = _normalize_text(record.get("kind")) or "unknown"

    if not user_input or not assistant_output:
        return None

    return {
        "kind": kind,
        "prompt": user_input,
        "response": assistant_output,
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output},
        ],
    }


def stratified_split_by_kind(
    rows: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for kind_rows in grouped.values():
        rng.shuffle(kind_rows)
        n = len(kind_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if n >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)

        train_rows.extend(kind_rows[:n_train])
        val_rows.extend(kind_rows[n_train : n_train + n_val])
        test_rows.extend(kind_rows[n_train + n_val :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def prepare_firefly_dataset(
    output_dir: str | Path,
    dataset_name: str = "YeungNLP/firefly-train-1.1M",
    dataset_split: str = "train",
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    raw_ds = load_dataset(dataset_name, split=dataset_split)
    if max_samples:
        raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))

    converted: list[dict[str, Any]] = []
    skipped = 0
    for row in raw_ds:
        item = convert_firefly_record(row)
        if item is None:
            skipped += 1
            continue
        converted.append(item)

    train_rows, val_rows, test_rows = stratified_split_by_kind(
        converted,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    write_jsonl(out / "test.jsonl", test_rows)

    metadata = {
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "total_raw": len(raw_ds),
        "total_kept": len(converted),
        "total_skipped": skipped,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "max_samples": max_samples,
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata
