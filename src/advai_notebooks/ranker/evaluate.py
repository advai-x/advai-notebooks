"""Score the provided candidates; no retrieval or positive injection."""

from advai_notebooks.common.retrieval import rerank

from .metrics import evaluate_rankings


def evaluate(model, data, cfg, split, candidates):
    evaluate_rankings(candidates, data, split, [1])
    rankings = rerank(model, data, candidates, cfg.training.batch_size, cfg.data.reject_truncation)
    return evaluate_rankings(rankings, data, split, cfg.evaluation.cutoffs), rankings
