# VLM configs

- `smoke.yaml`: tiny native LLaVA full update (frozen vision).
- `lora.yaml`: language q_proj/v_proj adapters on the tiny model.
- `hf.yaml`: fixed LLaVA 7B revision, LoRA template on your own image QA. High fp32 memory requirements; 7B GPU training is not validated.

All support single-process, single-device fp32. Model/data configuration is strict: unknown fields fail.
Use a fresh output directory for new experiments; `--resume` requires the original config/data/source/environment.
[Commands and field contracts](../../docs/vlm.md).
