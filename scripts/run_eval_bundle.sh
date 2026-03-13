#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash scripts/run_eval_bundle.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     outputs/h100_main/final_adapter \
#     data/firefly_prepared/test.jsonl \
#     outputs/h100_main/eval \
#     outputs/h100_main/trainer_state.json
#
# Evaluation runs on a single GPU.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

BASE_MODEL="${1:?missing base model name/path}"
ADAPTER_PATH="${2:?missing adapter path}"
TEST_FILE="${3:?missing test jsonl path}"
OUT_DIR="${4:?missing output directory}"
TRAINER_STATE="${5:-}"

mkdir -p "$OUT_DIR"

echo "=== Token-level NLL / Perplexity ==="
python scripts/evaluate_token_loss.py \
  --base-model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --test-file "$TEST_FILE" \
  --output-json "$OUT_DIR/token_loss_metrics.json" \
  --max-length 1024 \
  --batch-size 2 \
  --max-samples 2000

echo "=== Before / After Generation ==="
python scripts/generate_before_after.py \
  --base-model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --prompt-file prompts/eval_prompts_zh.jsonl \
  --output-jsonl "$OUT_DIR/before_after.jsonl" \
  --output-md "$OUT_DIR/before_after.md" \
  --max-new-tokens 256 \
  --temperature 0.0

if [[ -n "$TRAINER_STATE" ]]; then
  python scripts/plot_training_curves.py \
    --trainer-state "$TRAINER_STATE" \
    --output-png "$OUT_DIR/loss_curve.png"
fi
