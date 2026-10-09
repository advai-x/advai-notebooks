# Diffusion notebook

`01_finetune.ipynb` is an offline CPU tutorial using the same independent Python modules
as the CLI. It executes full/LoRA training, fixed-noise evaluation, export/reload, image
generation, LoRA merge and a publication dry-run. Each run writes to a fresh ignored
output directory. Install `.[diffusion,notebooks,publish]`; run from this repository.
The small synthetic images and random VAE/encoder verify lifecycle behavior only.
