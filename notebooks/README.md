# Notebooks

Use numbered notebooks for a repeatable lifecycle: `01_prepare_data`, `02_train`, `03_evaluate`, `04_inference`, and `05_export`. Shared implementation should be moved into `src/advai_notebooks`.

The first v1 notebooks each walk through the complete lifecycle in one file:

- [Embedding](embedding/01_finetune.ipynb): shared data, negatives, MNRL, retrieval metrics and export.
- [Ranker](ranker/01_finetune.ipynb): prepared pairs, fixed candidates, BCE and export.
- [Decision](decision/01_finetune.ipynb): native HF encoder + option scorer, CE/RLCD, calibration, metrics and export.

The optional [combined workflow](workflows/01_retrieval_comparison.ipynb) contains
the six-way comparison. Core task notebooks have executed offline CPU examples with saved cell outputs. Their
tiny synthetic models verify interfaces; they do not establish domain quality.

- [Diffusion](diffusion/01_finetune.ipynb): offline full/LoRA latent image training, fixed-noise evaluation, native reload, generated samples and an explicit publication preview.
