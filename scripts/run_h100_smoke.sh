#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_h100_smoke.sh
#
# Runs a 100-step smoke test on 2 GPUs (4,5) to validate setup.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

echo "=== Environment Check ==="
python scripts/check_environment.py

echo "=== Smoke Test (100 steps, 2 GPU DDP) ==="
accelerate launch \
  --config_file configs/accelerate_2gpu.yaml \
  scripts/train_oft_sft.py --config configs/h100_smoke100.yaml
