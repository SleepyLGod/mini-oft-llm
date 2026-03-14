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
#    7. Comparison plot
#
#  GPUs: CUDA devices 4 and 5 (configurable below).
#  Env : expects uv venv at .venv  (run `make setup` first).
###############################################################################

# --------------- configurable ------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export DRY_RUN="${DRY_RUN:-0}"
SESSION="oft-train"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_ACTIVATE="source ${REPO_ROOT}/.venv/bin/activate"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# -----------------------------------------------------------------------------

run_step() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] $*"
        return 0
    fi
    "$@"
}

dataset_ready() {
    [[ -f data/firefly_prepared/train.jsonl ]] \
      && [[ -f data/firefly_prepared/val.jsonl ]] \
      && [[ -f data/firefly_prepared/test.jsonl ]] \
      && [[ -f data/firefly_prepared/metadata.json ]]
}

train_ready() {
    local out_dir="$1"
    [[ -f "$out_dir/run_summary.json" ]] && [[ -d "$out_dir/final_adapter" ]]
}

eval_ready() {
    local train_dir="$1"
    local eval_dir="$2"
    [[ -f "$train_dir/run_summary.json" ]] \
      && [[ -f "$eval_dir/token_loss_metrics.json" ]] \
      && [[ -f "$eval_dir/before_after.md" ]] \
      && [[ -f "$eval_dir/loss_curve.png" ]] \
      && [[ "$eval_dir/token_loss_metrics.json" -nt "$train_dir/run_summary.json" ]] \
      && [[ "$eval_dir/before_after.md" -nt "$train_dir/run_summary.json" ]] \
      && [[ "$eval_dir/loss_curve.png" -nt "$train_dir/run_summary.json" ]]
}

comparison_ready() {
    [[ -f artifacts/loss_comparison.png ]] \
      && [[ -f outputs/h100_main/trainer_state.json ]] \
      && [[ -f outputs/h100_ablation_block16/trainer_state.json ]] \
      && [[ artifacts/loss_comparison.png -nt outputs/h100_main/trainer_state.json ]] \
      && [[ artifacts/loss_comparison.png -nt outputs/h100_ablation_block16/trainer_state.json ]]
}

# If already in the session, just run the pipeline directly (re-entrant).
if [[ "${TMUX:-}" == *"${SESSION}"* ]] || [[ "${RUN_INSIDE_TMUX:-}" == "1" ]]; then
    echo "========================================"
    echo " OFT Full Pipeline – GPU ${CUDA_VISIBLE_DEVICES}"
    echo "========================================"

    cd "$REPO_ROOT"
    $VENV_ACTIVATE

    echo ""
    echo ">>> [1/7] Dataset preparation"
    if dataset_ready; then
        echo "Skipping: dataset already prepared"
    else
        run_step python scripts/run_data_prep.py \
            --output-dir data/firefly_prepared \
            --dataset-name YeungNLP/firefly-train-1.1M \
            --dataset-split train \
            --train-ratio 0.9 \
            --val-ratio 0.05 \
            --seed 42
    fi

    echo ""
    echo ">>> [2/7] Smoke test (100 steps)"
    if train_ready outputs/h100_smoke100; then
        echo "Skipping: smoke run already complete"
    else
        run_step bash scripts/run_h100_smoke.sh
    fi

    echo ""
    echo ">>> [3/7] Main training (1200 steps)"
    if train_ready outputs/h100_main; then
        echo "Skipping: main run already complete"
    else
        echo "Running/resuming main run"
        run_step bash scripts/run_h100_main.sh
    fi

    echo ""
    echo ">>> [4/7] Ablation training (block_size=16, 1200 steps)"
    if train_ready outputs/h100_ablation_block16; then
        echo "Skipping: ablation run already complete"
    else
        echo "Running/resuming ablation run"
        run_step bash scripts/run_h100_ablation.sh
    fi

    echo ""
    echo ">>> [5/7] Evaluation – main run"
    if eval_ready outputs/h100_main outputs/h100_main/eval; then
        echo "Skipping: main eval already up to date"
    else
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
        run_step bash scripts/run_eval_bundle.sh \
            "$BASE_MODEL" \
            outputs/h100_main/final_adapter \
            data/firefly_prepared/test.jsonl \
            outputs/h100_main/eval \
            outputs/h100_main/trainer_state.json
    fi

    echo ""
    echo ">>> [6/7] Evaluation – ablation run"
    if eval_ready outputs/h100_ablation_block16 outputs/h100_ablation_block16/eval; then
        echo "Skipping: ablation eval already up to date"
    else
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
        run_step bash scripts/run_eval_bundle.sh \
            "$BASE_MODEL" \
            outputs/h100_ablation_block16/final_adapter \
            data/firefly_prepared/test.jsonl \
            outputs/h100_ablation_block16/eval \
            outputs/h100_ablation_block16/trainer_state.json
    fi

    echo ""
    echo ">>> [7/7] Comparison plot (main vs ablation)"
    if comparison_ready; then
        echo "Skipping: comparison plot already up to date"
    else
        run_step python scripts/plot_training_curves.py --compare \
            --trainer-state outputs/h100_main/trainer_state.json \
            --label "block_size=32 (main)" \
            --trainer-state outputs/h100_ablation_block16/trainer_state.json \
            --label "block_size=16 (ablation)" \
            --output-png artifacts/loss_comparison.png
    fi

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
    "export RUN_INSIDE_TMUX=1; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; export DRY_RUN=${DRY_RUN}; bash $0; exec bash"

echo ""
echo "Pipeline launched in tmux session '$SESSION'."
if [[ "$DRY_RUN" == "1" ]]; then
    echo "(DRY_RUN=1 enabled)"
fi
echo ""
echo "  Attach:   tmux attach -t $SESSION"
echo "  Detach:   Ctrl-b d"
echo "  Kill:     tmux kill-session -t $SESSION"
echo ""

