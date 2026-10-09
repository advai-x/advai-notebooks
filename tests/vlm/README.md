# VLM tests

`test_vlm_native.py` covers task-specific model behavior, native exports and inference.
Shared lifecycle regressions are in [test_multimodal_lifecycle.py](../test_multimodal_lifecycle.py),
including exact interrupted epoch recovery, offline whitelist publication and independent imports.

```bash
python -m pytest tests/vlm tests/test_multimodal_lifecycle.py -q
```

Tests use random tiny models and local synthetic data; GPU and real task quality are not implied.
