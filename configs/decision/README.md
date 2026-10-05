# Decision configs

Decision-model training, calibration, evaluation and inference configuration. Model-specific settings belong in individual config filenames rather than subdirectories.

`training.optimizer` selects `adamw` (default) or `adam`; encoder/head retain
separate learning rates. `evaluation.selection_metric` supports `nll`, `brier`,
`ece` and `accuracy`, automatically selecting the appropriate direction.
`evaluation.include_baseline` defaults to true; epoch zero participates in selection.
`calibration.enabled` defaults to true and requires a separate calibration split.
Set it to false to use train/validation/test only and export uncalibrated probabilities.

Resume the original run using `run --config <config> --resume <output_dir>/latest`.
Keep the original configuration and planned epoch count unchanged. See the
[architecture guide](../../docs/architecture.md) for checkpoint boundaries.

Decision loads HF encoders directly in `model.py`, which also saves/reloads
native exports. No separate provider layer is used. `model.provider: hf` remains
a compatible backend identifier; set `model.id` to a compatible base encoder/local
encoder directory. `hf.yaml` is the multilingual BERT example. Pooling (`mean` or
`cls`), head width/dropout and encoder freezing are independently configurable.
Temperature minimum counts are `calibration.min_type_samples` and
`calibration.min_bucket_samples`. The scorer starts randomly over a pretrained
encoder. Old SDK model/config formats have been removed and need retraining.

## Self-owned components and objectives

The supported components are `model.encoding: pair`, `model.head: independent`,
and `model.decoder: typed`. They have separate interfaces/implementations;
unsupported names fail explicitly. HF is the base encoder loader, and our own
head starts randomly. See the [Decision architecture](../../docs/architecture.md)
for contracts, shapes and extension requirements.

Default `training.loss: supervised` uses soft CE for every type and normalized
ranked probability score for `score`, weighted by `ce_weight` and `ordinal_weight`.
`loss: ce` provides a pure CE control; `loss: rlcd` adds optional native
Gaussian-logit policy gradients to the supervised objective. Set `ordinal_weight: 0`
to reproduce the earlier CE + RLCD combination. RLCD is an experiment, not a
prerequisite for a decision model.

`smoke.yaml`, `ce_smoke.yaml` and `rlcd_smoke.yaml` use the same tiny model, data,
seed and training/selection/calibration budget with separate outputs. They check
the pipeline; use real domain data and multiple seeds to compare effectiveness.
Native schema v1 inference exports migrate automatically to v2; strict resume
requires unchanged implementation and does not migrate earlier training states.
