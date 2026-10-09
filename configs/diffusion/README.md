# Diffusion configuration

- `smoke.yaml`: offline random tiny latent pipeline, full UNet training.
- `lora.yaml`: the same fixture pipeline, UNet-only LoRA; alpha differs from rank to verify serialization.
- `hf.yaml`: configurable Stable Diffusion v1/v2-style base and real image/caption JSONL.
- `prompts.json`: CLI generation input.

Run from the repository root; install `.[diffusion]`. All output paths are independent.
The first backend is single-device fp32. Full means UNet only; VAE/text encoder stay frozen.
SDXL/DiT/video and mixed precision are unsupported. See [the guide](../../docs/diffusion.md).
