#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_h100_ablation.sh

python scripts/train_oft_sft.py --config configs/h100_ablation_block16.yaml
