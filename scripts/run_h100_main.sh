#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_h100_main.sh
#
# Assumes you are in repo root and dependencies are installed.

python scripts/check_environment.py
python scripts/train_oft_sft.py --config configs/h100_main.yaml
