# VLA outputs

Generated files are ignored by Git. Each run records:

- `config.json`, `environment.json`, `data_summary.json` and baseline metrics.
- `checkpoints/checkpoint-N/`: model, optimizer/RNG and epoch state.
- `latest/state.json`: atomic pointer to the latest completed checkpoint.
- `history.json` and `report.json`: validation selection and report-only test results.
- `export/`: best validation model, including epoch zero if it wins.
- `latest_export/`: last trained model.

Keep this directory on persistent storage in Colab. Never publish an entire run directory;
use `publish --export-dir .../export --dry-run` to inspect the inference-only package.
[Full guide](../../docs/vla.md).
