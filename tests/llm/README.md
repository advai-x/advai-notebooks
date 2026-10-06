# LLM tests

Run `python -m pytest tests/llm -q` after installing both LLM extras and dev dependencies.
Checks cover CLM/SFT supervision and split integrity, full/LoRA training, token-weighted
evaluation, native reload/merge, publication contracts, CLI, task independence,
gradient accumulation and exact epoch recovery on CPU/fp32.
