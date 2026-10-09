# VLA configs

- `smoke.yaml`: tiny CLIP, frozen encoder and trainable action head.
- `full.yaml`: update the tiny encoder and action head together.
- `hf.yaml`: fixed pretrained CLIP revision + random action head on your own episodes; not OpenVLA.

All support single-process, single-device fp32. Model/data configuration is strict: unknown fields fail.
Use a fresh output directory for new experiments; `--resume` requires the original config/data/source/environment.
[Commands and field contracts](../../docs/vla.md).
