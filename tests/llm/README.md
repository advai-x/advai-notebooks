# LLM tests

Run `python -m pytest tests/llm -q` after installing both LLM extras and dev dependencies.
Checks cover CLM/SFT supervision and split integrity, full/LoRA training, token-weighted
evaluation, native reload/merge, publication contracts, CLI, task independence,
gradient accumulation and exact epoch recovery on CPU/fp32.

`test_moe.py` adds native tiny OLMoE / Qwen3-MoE CPT/SFT with full,
attention LoRA and native 3D expert-parameter LoRA. It checks actual router/expert
gradients, frozen base weights, separate auxiliary loss, padding-aware routing,
resume equality, native merge/reload, storage dtype and offline publication.
Real pretrained GPU training and domain quality are not part of this suite.
