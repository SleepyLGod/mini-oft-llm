#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_h100_main.sh
#
# Main OFT training run on 2 GPUs (4,5) with DDP.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

echo "=== Environment Check ==="
python scripts/check_environment.py

echo "=== Main Training (1200 steps, 2 GPU DDP) ==="
accelerate launch \
  --config_file configs/accelerate_2gpu.yaml \
  scripts/train_oft_sft.py --config configs/h100_main.yaml
