# Report Data Guide

> 本文件是写报告前的"导航手册"。  
> 所有关键数字、产物路径和写作建议都在这里。

---

## 一、实验完成状态

| 阶段 | 状态 | 关键产物 |
|------|------|----------|
| 数据准备 | ✅ 完成 | `data/firefly_prepared/metadata.json` |
| Smoke test (100 steps) | ✅ 完成 | `outputs/h100_smoke100/run_summary.json` |
| Main OFT run (1200 steps, block_size=32) | ✅ 完成 | `outputs/h100_main/run_summary.json` |
| Ablation run (1200 steps, block_size=16) | ✅ 完成 | `outputs/h100_ablation_block16/run_summary.json` |
| Main 外部评估 | ✅ 完成 | `outputs/h100_main/eval/` |
| Ablation 外部评估 | ✅ 完成 | `outputs/h100_ablation_block16/eval/` |
| Main vs Ablation 对比图 | ✅ 完成 | `artifacts/loss_comparison.png` |

---

## 二、关键定量结果

### 2.1 Token-level NLL / Perplexity（外部评估，2000 条测试样本）

| 模型 | Mean NLL | Perplexity |
|------|----------|------------|
| Base（未微调） | 3.2619 | **26.099** |
| OFT block_size=32（main） | 1.9770 | **7.221** |
| OFT block_size=16（ablation） | 1.9988 | **7.380** |

- **Main vs Base Δ NLL**: −1.2849（↓ 39.4%）  
- **Ablation vs Base Δ NLL**: −1.2631（↓ 38.7%）  
- **Main vs Ablation**: main 略优，PPL 低 0.159

来源文件：  
- `outputs/h100_main/eval/token_loss_metrics.json`  
- `outputs/h100_ablation_block16/eval/token_loss_metrics.json`

### 2.2 Trainer 内部 eval_loss（训练结束时 validation set）

| Run | eval_loss |
|-----|-----------|
| Main (block_size=32) | 1.7879 |
| Ablation (block_size=16) | 1.8102 |

来源文件：  
- `outputs/h100_main/run_summary.json`  
- `outputs/h100_ablation_block16/run_summary.json`

> **⚠️ 注意**：`run_summary.json` 中的 `train_loss` 数值不具可比性。  
> Main run 曾中断后从 checkpoint-1000 续跑，其 `train_loss=0.321` 仅反映最后 ~200 步的均值；  
> Ablation 从头跑完，`train_loss=1.985` 是全程均值（训练初期 loss 高会拉高均值）。  
> **报告中应使用 `eval_loss` 和外部 PPL，而非 `train_loss`**。

---

## 三、报告可用素材路径

### 图表
| 素材 | 路径 | 建议放在报告哪里 |
|------|------|-----------------|
| Main loss curve | `outputs/h100_main/eval/loss_curve.png` | Training 小节 |
| Ablation loss curve | `outputs/h100_ablation_block16/eval/loss_curve.png` | Ablation 小节 |
| Main vs Ablation 对比图 | `artifacts/loss_comparison.png` | Ablation 小节（核心图） |

### 定性样例
| 素材 | 路径 |
|------|------|
| Main before/after（Markdown 格式） | `outputs/h100_main/eval/before_after.md` |
| Ablation before/after（Markdown 格式） | `outputs/h100_ablation_block16/eval/before_after.md` |
| Main before/after（JSONL，可程序处理） | `outputs/h100_main/eval/before_after.jsonl` |

### 模型权重（如需本地推理验证）
| 素材 | 路径 |
|------|------|
| Main final adapter | `outputs/h100_main/final_adapter/` |
| Ablation final adapter | `outputs/h100_ablation_block16/final_adapter/` |

---

## 四、实验设置摘要（报告 Method 小节参考）

| 参数 | 值 |
|------|----|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Dataset | `YeungNLP/firefly-train-1.1M`（中文指令微调） |
| Train split | 90%（约 148 万条） |
| Val / Test split | 各 5%（约 8.2 万条） |
| 微调方法 | OFT（Orthogonal Fine-Tuning）via PEFT |
| Training steps | 1200 |
| Hardware | 2× NVIDIA H100 NVL（DDP） |
| Precision | bf16 |
| Main: OFT block_size | 32 |
| Ablation: OFT block_size | 16 |

---

## 五、报告写作建议

### 5.1 各小节核心内容

**Introduction**  
- 介绍 OFT 的动机：在微调时保持权重空间的正交性，避免灾难性遗忘  
- 说明选用 Qwen2.5-7B 和中文指令数据集的理由

**Method**  
- 从 Section 四的表格直接引用实验设置  
- 重点说明 OFT 与 LoRA 的区别（正交矩阵约束 vs 低秩分解）

**Results**  
- 主表：引用 Section 2.1 的 NLL / PPL 数字  
- 放入 `loss_curve.png` 展示收敛过程

**Ablation Study**  
- 放入 `artifacts/loss_comparison.png`（main vs ablation 同一张图）  
- 结论：block_size=32 比 block_size=16 略优（PPL 7.221 vs 7.380）  
- 原因分析：更大的 block_size 在正交矩阵中保留了更完整的子空间结构

**Qualitative Analysis**  
- 从 `before_after.md` 挑 3–5 个典型例子  
- 建议选：before 回答明显不流畅/错误，after 明显改善的样本

**Conclusion**  
- OFT 实现了显著 PPL 下降（26.1 → 7.2，↓72%）  
- block_size=32 为更优配置  
- 局限性：训练 loss 因续跑导致数值不直接可比（见 Section 2.2 注意）

### 5.2 报告中不要用的数字

- `run_summary.json` 中的 `train_loss`（原因见 Section 2.2 注意）  
- `train_samples_per_second` / `train_steps_per_second`（两次运行环境不同）

---

## 六、数据未上传说明（.gitignore）

以下文件因体积过大被排除在 Git 之外：

| 文件/目录 | 原因 |
|-----------|------|
| `data/firefly_prepared/*.jsonl` | train=2.1 GB，test/val 各 115 MB |
| `outputs/*/checkpoint-*/` | 含 optimizer.pt，每个约 200–400 MB |

如需在本地重现数据集，运行：
```bash
uv run python scripts/run_data_prep.py \
  --output-dir data/firefly_prepared \
  --dataset-name YeungNLP/firefly-train-1.1M \
  --dataset-split train \
  --train-ratio 0.9 --val-ratio 0.05 --seed 42
```

