#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_h100_ablation.sh
#
# Ablation run (block_size=16) on 2 GPUs (4,5) with DDP.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

echo "=== Ablation Training (block_size=16, 1200 steps, 2 GPU DDP) ==="
accelerate launch \
  --config_file configs/accelerate_2gpu.yaml \
  scripts/train_oft_sft.py --config configs/h100_ablation_block16.yaml
