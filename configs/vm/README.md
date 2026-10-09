# VM configs

- `smoke.yaml`: random residual MLP on numerical state/action episodes.
- `finetune.yaml`: first run smoke, then fine-tune its native export into a new output directory. Preserves normalization and state semantics.

All support single-process, single-device fp32. Model/data configuration is strict: unknown fields fail.
Use a fresh output directory for new experiments; `--resume` requires the original config/data/source/environment.
[Commands and field contracts](../../docs/vm.md).
