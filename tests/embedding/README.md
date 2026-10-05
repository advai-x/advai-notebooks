# Embedding tests

Run `python -m pytest tests/embedding -q`. Offline CPU tests cover canonical
data validation, group leakage, known-positive exclusion, exact chunked
retrieval, graded metrics, both MNRL objectives and native SDK exports.

Component checks additionally cover mean/CLS pooling, normalization, learned
Dense tails and their gradients, parameter names, and native export reloads.

`tests/test_independent_training.py` additionally forbids ranker imports in a
fresh process while running embedding training, export reload, inference and CLI.
