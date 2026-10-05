# Independent ranker configs

`smoke.yaml` reads prepared synthetic labeled pairs and candidates, and trains a
random tiny classifier without embedding. `bge.yaml` uses BGE reranker and
user-prepared artifacts. Change `data.training_samples` and `data.candidates`
independently of embedding settings. Ranker has no `retrieval` section.
See [the ranker guide](../../notebooks/ranker/README.md).

`training.optimizer` selects `adamw` (default) or `adam`. Task-specific parameter
groups can be customized in this task's `optimizer.py`. Completed epoch state is
saved under `latest/training_state.pt`; recover an interrupted run with
`run --config <config> --resume <output_dir>/latest`, keeping the original
configuration and planned epoch count. Native `export/` remains inference-only.
