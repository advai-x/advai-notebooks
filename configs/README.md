# Configs

Store versioned YAML experiment configurations here. Keep credentials, machine-specific overrides and generated state outside Git.

Each task has an offline `smoke.yaml`. Larger model presets are templates: replace the data/output paths, inspect memory requirements and validate before training.

- [Diffusion](diffusion/README.md): full/LoRA and Stable Diffusion.
- [VLM](vlm/README.md): native LLaVA full/LoRA; fixed 7B Hub revision template.
- [VLA](vla/README.md): CLIP continuous-action head; frozen/full encoder updates.
- [VM / World Model](vm/README.md): state dynamics and fine-tuning a previous export.
