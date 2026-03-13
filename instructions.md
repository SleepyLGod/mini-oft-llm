# Mini-Project: Parameter-efficient Finetuning for Pretrained Foundation Models

**Task:** 
Use orthogonal finetuning (OFT) to finetune a pretrained model (eg, Stable Diffusion, Llama, Qwen, or any pretrained models) for any downstream task and summarize your experimental results and findings [file:1].

**Deadline:** 
22th March 23:59 pm, 2026 [file:1]

**Deliverables:**
- Project code (attached as a Github repository link and a Readme in the repo) [file:1]
- Project report (3 pages): it should include your training loss curves and the final task performance and/or qualitative results (before and after the finetuning) [file:1]

**Tool:**
- OFT: https://huggingface.co/docs/peft/main/en/conceptual_guides/oft [file:1]

**Potential downstream task:**
- Subject-driven generation (finetune a pretrained diffusion model with your own images) [file:1]:
  - Hugging Face OFT example: https://github.com/huggingface/peft/blob/main/examples/boft_dreambooth/boft_dreambooth.md [file:1]
  - OFT example: https://github.com/zqiu24/oft [file:1]
- Many more example tasks can be found in https://github.com/huggingface/peft/tree/main/examples [file:1]
- If you don’t have sufficient GPUs, it is ok to finetune a smaller model, say Qwen-1.3B [file:1]

