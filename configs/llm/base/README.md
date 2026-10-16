# LLM base configs

- `smoke.yaml`: offline tiny decoder, full fine-tuning on synthetic fixtures.
- `lora.yaml`: offline tiny decoder, LoRA with a portable bundled frozen base.
- `hf.yaml`: configurable real-model LoRA example; replace the dataset and output directory.

Instruct requires a fast tokenizer and an aligned chat template. Complex templates
whose earlier turns change when more messages are appended need explicit generation blocks.
Models/hardware are configurable; real-model and CUDA/MPS training require validation
in the target environment. QLoRA, packing, multi-device and preference training are not implemented.

`moe/` adds native OLMoE / Qwen3-MoE smoke and pinned real-model presets.
See [MoE configuration and memory limits](../../../docs/llm-moe.md).
The existing task's data/objective stay unchanged; MoE is not a separate task.
