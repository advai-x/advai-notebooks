# Ranker tests

Run `python -m pytest tests/ranker -q`. Offline CPU tests cover real BCE weight
updates, pair sequence budgets, fixed candidate membership, missing-positive
recall, native SDK scores and the combined embedding/ranker comparison.

Component checks cover BERT/RoBERTa encoder/classifier order, stochastic dropout
and gradient agreement, parameter names, and native export reloads.

`tests/test_independent_training.py` additionally forbids embedding imports in a
fresh process while running ranker training, export reload, inference and CLI.
