# Notebooks

English | [简体中文](README.zh-CN.md)

Core tutorials explain the same implementation used by the CLI. Each `01_finetune.ipynb` walks through validation, training, evaluation, export/reload and inference.

- [Embedding](embedding/01_finetune.ipynb): MNRL and retrieval evaluation.
- [Ranker](ranker/01_finetune.ipynb): BCE and held-out ranking.
- [Decision](decision/01_finetune.ipynb): native encoder, option scoring and calibration.
- [LLM Base](llm/base/01_finetune.ipynb) / [Instruct](llm/instruct/01_finetune.ipynb): continued pre-training and assistant SFT.
- [Diffusion](diffusion/01_finetune.ipynb): latent text-to-image full/LoRA.
- [VLM](vlm/01_finetune.ipynb): native LLaVA image QA, full/LoRA and multimodal processor reload.
- [VLA](vla/01_finetune.ipynb): CLIP + state continuous action chunks, episode masks and physical units.
- [VM](vm/01_finetune.ipynb): numerical World Model dynamics, native fine-tuning and recursive rollouts.

The [optional retrieval comparison](workflows/01_retrieval_comparison.ipynb) combines independently trained embedding and ranker models.
Tiny random models and synthetic CPU examples validate interfaces, not real-world quality.
Colab cells install task-specific extras; persistent data/checkpoint storage is required for longer experiments.

English is the default. Each tutorial has a `*.zh-CN.ipynb` companion with a language
switch in its opening cell. Both versions use the same executable workflow; model
inputs and language-specific prefixes are not translated as documentation.
