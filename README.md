# OFT Mini Project: Chinese Instruction Tuning with a Small LLM

This repository implements the mini-project requirement: **parameter-efficient finetuning with OFT** on a pretrained foundation model for a downstream task.

- Method: **OFT (Orthogonal Finetuning)** via Hugging Face PEFT
- Task: **Chinese instruction-following SFT**
- Base model: **Qwen/Qwen2.5-7B-Instruct**
- Training plan: H100 (smoke + main + ablation)

## 1) Repository Layout

```text
mini-oft-llm/
├── configs/
│   ├── base.yaml
│   ├── h100_smoke100.yaml
│   ├── h100_main.yaml
│   └── h100_ablation_block16.yaml
├── prompts/
│   └── eval_prompts_zh.jsonl
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
│   └── run_eval_bundle.sh
├── src/mini_oft_llm/
│   ├── config.py
│   ├── data.py
│   ├── training.py
│   └── eval.py
├── requirements.txt
└── README.md
```

## 2) Environment Setup (Remote Linux GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If your CUDA image requires a specific torch build, install torch first and then run `pip install -r requirements.txt`.

For Apple Silicon Mac local development, use this repo for preparation only (config/scripts/prompt/report work). Training is expected on remote Linux + NVIDIA GPUs.

## 3) Step-by-Step Execution

### Step A. Prepare Dataset (Run once)

```bash
python scripts/run_data_prep.py \
  --output-dir data/firefly_prepared \
  --dataset-name YeungNLP/firefly-train-1.1M \
  --dataset-split train \
  --train-ratio 0.9 \
  --val-ratio 0.05 \
  --seed 42
```

Optional fast iteration mode:

```bash
python scripts/run_data_prep.py --output-dir data/firefly_prepared --max-samples 120000
```

### Step B. H100 Smoke Run (Recommended)

```bash
bash scripts/run_h100_smoke.sh
```

This is a fast 100-step validation pass before committing full H100 time.

### Step C. H100 Main Run

```bash
bash scripts/run_h100_main.sh
```

### Step D. H100 Ablation Run

```bash
bash scripts/run_h100_ablation.sh
```

Current ablation changes only one factor: `oft_block_size` (`32 -> 16`).

### Step E. Evaluate and Export Artifacts

```bash
bash scripts/run_eval_bundle.sh \
  Qwen/Qwen2.5-7B-Instruct \
  outputs/h100_main/final_adapter \
  data/firefly_prepared/test.jsonl \
  outputs/h100_main/eval \
  outputs/h100_main/trainer_state.json
```

This generates:
- token-level NLL / perplexity comparison (`base vs OFT`)
- fixed-prompt before/after generations
- training curve figure from `trainer_state.json`

## 4) Config Notes

- `configs/h100_main.yaml`: full main experiment
- `configs/h100_smoke100.yaml`: preflight smoke on H100
- `configs/h100_ablation_block16.yaml`: single-factor ablation

Key OFT fields are under `oft:`.

## 5) Expected Outputs

Typical files after runs:

- `outputs/h100_main/final_adapter/` (trained OFT adapter)
- `outputs/h100_main/trainer_state.json` (for plotting loss curves)
- `outputs/h100_main/eval/token_loss_metrics.json`
- `outputs/h100_main/eval/before_after.jsonl`
- `outputs/h100_main/eval/before_after.md`
- `outputs/h100_main/eval/loss_curve.png`

## 6) Report Checklist (3 Pages, English)

Use `report/REPORT_TEMPLATE.md` and include:

1. Setup and method (OFT configuration)
2. Loss curves and quantitative metrics
3. Before/after qualitative examples
4. One concise ablation analysis
5. Reproducibility details (configs and commands)

## 7) Assignment Alignment

This repo explicitly satisfies:
- OFT-based PEFT on a pretrained model
- downstream task finetuning and reporting
- training loss curve
- final performance and/or qualitative before-vs-after results

## 8) Reproducibility Tips

- Keep dataset split seed fixed (`seed=42`)
- Keep prompt set fixed (`prompts/eval_prompts_zh.jsonl`)
- Save `resolved_config.yaml` per run (automatically done by training script)
- Run at least one smoke before main run when switching machines
