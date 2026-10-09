# Diffusion tests

Run `pytest tests/diffusion -q`. CPU/offline tests use real tiny Diffusers components.
Checks cover full/LoRA updates and frozen components, rank/alpha-preserving native reload,
merged exports, data leakage, epsilon/velocity targets, sample-weighted accumulation,
fixed-noise evaluation, exact interrupted epoch recovery and strict resume rejection,
CLI generation, offline publication and imports blocked for all other tasks.
