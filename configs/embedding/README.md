# Embedding configs

`smoke.yaml` runs an offline tiny BERT on the shared synthetic retrieval fixture.
`bge.yaml` shows a configurable real Chinese BGE encoder, prompts and Cached MNRL.
Data paths are relative to the project root; absolute paths work too. Hub models
can pin `revision`; local checkpoints use `model.id` paths. Outputs must be fresh for a new run.
See [the training guide](../../notebooks/embedding/README.md) for all settings.

`data.training_samples` selects prepared contrastive rows. Negative mining settings
live in `configs/data_preparation/`, separately from embedding training.

`training.optimizer` selects `adamw` (default) or `adam`. Task-specific parameter
groups can be customized in this task's `optimizer.py`. Completed epoch state is
saved under `latest/training_state.pt`; recover an interrupted run with
`run --config <config> --resume <output_dir>/latest`, keeping the original
configuration and planned epoch count. Native `export/` remains inference-only.
