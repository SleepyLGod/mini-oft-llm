# Mini-Project Report Template (English, 3 pages)

## Title
**Parameter-Efficient Finetuning with OFT for Chinese Instruction Following**

## 1. Task Definition
- Downstream task and motivation
- Why OFT for this task
- Base model and dataset choices

## 2. Experimental Setup
- Hardware (H100 96GB)
- Software stack and key package versions
- Dataset split protocol (`train/val/test`, fixed seed)
- OFT hyperparameters (`target_modules`, `oft_block_size`, `use_cayley_neumann`, etc.)

## 3. Main Results
- Training and validation loss curves
- Quantitative comparison (e.g., token-level NLL/perplexity)
- Table: Base model vs OFT model

## 4. Qualitative Analysis
- Fixed prompt set before/after comparison
- At least 6 representative examples from different groups
- Short discussion of improvements and remaining errors

## 5. Ablation Study
- One-factor ablation only (e.g., `oft_block_size=32` vs `16`)
- Explain impact on quality/stability/convergence

## 6. Conclusion
- Key findings in 3-5 bullet points
- Practical limitations
- Next-step improvements under a fixed compute budget

## 7. Reproducibility Appendix (brief)
- Command lines used for main run and ablation
- Config file names and artifact paths
- Random seed and split method
