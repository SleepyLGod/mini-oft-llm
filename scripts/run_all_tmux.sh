#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#  Master script – runs the full OFT pipeline inside a tmux session.
#
#  Usage (from repo root):
#      bash scripts/run_all_tmux.sh
#
#  What it does (sequentially inside tmux):
#    1. Prepare dataset
#    2. Smoke test  (2 GPU DDP, ~5 min)
#    3. Main run    (2 GPU DDP, ~2-3 h)
#    4. Ablation    (2 GPU DDP, ~2-3 h)
#    5. Eval bundle for main
#    6. Eval bundle for ablation
#
#  GPUs: CUDA devices 4 and 5 (configurable below).
#  Env : expects uv venv at .venv  (run `make setup` first).
###############################################################################

# --------------- configurable ------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
SESSION="oft-train"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_ACTIVATE="source ${REPO_ROOT}/.venv/bin/activate"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# -----------------------------------------------------------------------------

# If already in the session, just run the pipeline directly (re-entrant).
if [[ "${TMUX:-}" == *"${SESSION}"* ]] || [[ "${RUN_INSIDE_TMUX:-}" == "1" ]]; then
    echo "========================================"
    echo " OFT Full Pipeline – GPU ${CUDA_VISIBLE_DEVICES}"
    echo "========================================"

    cd "$REPO_ROOT"
    $VENV_ACTIVATE

    echo ""
    echo ">>> [1/6] Dataset preparation"
    python scripts/run_data_prep.py \
        --output-dir data/firefly_prepared \
        --dataset-name YeungNLP/firefly-train-1.1M \
        --dataset-split train \
        --train-ratio 0.9 \
        --val-ratio 0.05 \
        --seed 42

    echo ""
    echo ">>> [2/6] Smoke test (100 steps)"
    bash scripts/run_h100_smoke.sh

    echo ""
    echo ">>> [3/6] Main training (1200 steps)"
    bash scripts/run_h100_main.sh

    echo ""
    echo ">>> [4/6] Ablation training (block_size=16, 1200 steps)"
    bash scripts/run_h100_ablation.sh

    echo ""
    echo ">>> [5/6] Evaluation – main run"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
    bash scripts/run_eval_bundle.sh \
        "$BASE_MODEL" \
        outputs/h100_main/final_adapter \
        data/firefly_prepared/test.jsonl \
        outputs/h100_main/eval \
        outputs/h100_main/trainer_state.json

    echo ""
    echo ">>> [6/6] Evaluation – ablation run"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
    bash scripts/run_eval_bundle.sh \
        "$BASE_MODEL" \
        outputs/h100_ablation_block16/final_adapter \
        data/firefly_prepared/test.jsonl \
        outputs/h100_ablation_block16/eval \
        outputs/h100_ablation_block16/trainer_state.json

    echo ""
    echo ">>> [7/7] Comparison plot (main vs ablation)"
    python scripts/plot_training_curves.py --compare \
        --trainer-state outputs/h100_main/trainer_state.json \
        --label "block_size=32 (main)" \
        --trainer-state outputs/h100_ablation_block16/trainer_state.json \
        --label "block_size=16 (ablation)" \
        --output-png artifacts/loss_comparison.png

    echo ""
    echo "========================================"
    echo " ALL DONE.  Check outputs/ for results."
    echo "========================================"
    exit 0
fi

# --------------- Launch inside a new tmux session ----------------------------
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with:"
    echo "  tmux attach -t $SESSION"
    exit 1
fi

echo "Creating tmux session: $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "export RUN_INSIDE_TMUX=1; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; bash $0; exec bash"

echo ""
echo "Pipeline launched in tmux session '$SESSION'."
echo ""
echo "  Attach:   tmux attach -t $SESSION"
echo "  Detach:   Ctrl-b d"
echo "  Kill:     tmux kill-session -t $SESSION"
echo ""

