# OFT Mini Project: Chinese Instruction Tuning with Qwen2.5-7B

Parameter-efficient finetuning with **OFT (Orthogonal Finetuning)** on a pretrained foundation model for a downstream task.

| Item | Value |
|------|-------|
| Method | OFT via Hugging Face PEFT (`peft.OFTConfig`) |
| Task | Chinese instruction-following SFT |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Dataset | `YeungNLP/firefly-train-1.1M` (Chinese) |
| Hardware | 2 × NVIDIA H100 (GPU 4 & 5) |
| Training | `accelerate` DDP, 2-GPU |

## Repository Layout

```text
mini-oft-llm/
├── configs/
│   ├── base.yaml                    # base config template
│   ├── h100_smoke100.yaml           # 100-step smoke test
│   ├── h100_main.yaml               # main experiment
│   ├── h100_ablation_block16.yaml   # ablation (block_size=16)
│   └── accelerate_2gpu.yaml         # accelerate: 2-GPU DDP
├── prompts/
│   └── eval_prompts_zh.jsonl        # 50 fixed eval prompts
├── report/
│   └── REPORT_TEMPLATE.md
├── scripts/
│   ├── check_environment.py
│   ├── run_data_prep.py
│   ├── train_oft_sft.py
│   ├── evaluate_token_loss.py
│   ├── generate_before_after.py
│   ├── plot_training_curves.py
│   ├── run_h100_smoke.sh
│   ├── run_h100_main.sh
│   ├── run_h100_ablation.sh
│   ├── run_eval_bundle.sh
│   └── run_all_tmux.sh             # ★ one-click full pipeline (tmux)
├── src/mini_oft_llm/
│   ├── config.py
│   ├── data.py
│   ├── training.py
│   └── eval.py
├── pyproject.toml                   # project deps (managed by uv)
└── README.md
```

## Quick Start (One Command)

```bash
# 1. Create env with uv
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync --extra dev

# 2. Launch the full pipeline in a tmux session (GPU 4,5)
bash scripts/run_all_tmux.sh
```

This runs **everything** end-to-end inside tmux: data prep → smoke → main → ablation → eval.

Attach / detach:
```bash
tmux attach -t oft-train   # attach
# Ctrl-b d                 # detach (training continues)
tmux kill-session -t oft-train  # kill
```

## Environment Setup

### Option A – uv (recommended)

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
```

### Option B – plain pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

> If your CUDA image requires a specific torch build, install torch first.

## Step-by-Step Manual Execution

All training scripts default to `CUDA_VISIBLE_DEVICES=4,5` and use 2-GPU DDP via `accelerate`.
Override with `export CUDA_VISIBLE_DEVICES=...` if needed.

### Step A. Prepare Dataset (run once)

```bash
python scripts/run_data_prep.py \
  --output-dir data/firefly_prepared \
  --dataset-name YeungNLP/firefly-train-1.1M \
  --dataset-split train \
  --train-ratio 0.9 \
  --val-ratio 0.05 \
  --seed 42
```

### Step B. Smoke Test (~5 min)

```bash
bash scripts/run_h100_smoke.sh
```

### Step C. Main Training (~2-3 h)

```bash
bash scripts/run_h100_main.sh
```

### Step D. Ablation Training (~2-3 h)

```bash
bash scripts/run_h100_ablation.sh
```

Ablation changes one factor: `oft_block_size` (32 → 16).

### Step E. Evaluate and Export Artifacts

```bash
bash scripts/run_eval_bundle.sh \
  Qwen/Qwen2.5-7B-Instruct \
  outputs/h100_main/final_adapter \
  data/firefly_prepared/test.jsonl \
  outputs/h100_main/eval \
  outputs/h100_main/trainer_state.json
```

Generates: NLL/perplexity comparison, before/after text, loss curve plot.

## Config Notes

| Config | Purpose |
|--------|---------|
| `configs/h100_main.yaml` | Main experiment (block_size=32, 1200 steps) |
| `configs/h100_smoke100.yaml` | Preflight smoke (100 steps) |
| `configs/h100_ablation_block16.yaml` | Ablation (block_size=16) |
| `configs/accelerate_2gpu.yaml` | 2-GPU DDP with bf16 (training) |

OFT hyperparameters are under the `oft:` section in each config.

## Expected Outputs

```
outputs/h100_main/
├── final_adapter/              # trained OFT adapter
├── trainer_state.json          # for loss curve plotting
├── resolved_config.yaml        # snapshot of config used
├── run_summary.json            # train + eval metrics
└── eval/
    ├── token_loss_metrics.json # NLL / perplexity (base vs OFT)
    ├── before_after.jsonl      # generation comparison
    ├── before_after.md         # human-readable comparison
    └── loss_curve.png          # train / eval loss plot
```

## Report (3 Pages, English)

Use `report/REPORT_TEMPLATE.md` as a starting point. Include:

1. Setup and method (OFT configuration)
2. Training / validation loss curves
3. Quantitative metrics (NLL, perplexity)
4. Before/after qualitative examples
5. Ablation analysis (block_size 32 vs 16)
6. Reproducibility details

## Assignment Alignment

This repo satisfies all requirements from `instructions.md`:

- ✅ OFT-based PEFT on a pretrained model (Qwen2.5-7B)
- ✅ Downstream task finetuning (Chinese instruction following)
- ✅ Training loss curves
- ✅ Final performance (NLL/perplexity) and qualitative before-vs-after results
- ✅ GitHub repo with README
- ✅ 3-page report template included

## Reproducibility

- Dataset split seed: `42`
- Eval prompts: `prompts/eval_prompts_zh.jsonl` (50 fixed prompts)
- Each run saves `resolved_config.yaml` automatically
- Run smoke test first when switching machines
