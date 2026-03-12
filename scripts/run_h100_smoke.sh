#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_h100_smoke.sh

python scripts/check_environment.py
python scripts/train_oft_sft.py --config configs/h100_smoke100.yaml
