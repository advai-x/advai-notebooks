# Tests

Fast unit, contract and tiny CPU lifecycle tests; no expensive real-model downloads are needed.

```bash
python -m pytest tests/test_multimodal_contracts.py tests/test_multimodal_lifecycle.py tests/vlm tests/vla tests/vm -q
```

The three new tasks cover data/media leakage guards, normalization, padding/answer masks,
native export/reload, inference, exact interrupted epoch recovery, offline publication and task isolation.
GPU and long-running model-quality checks are outside the smoke suite.
