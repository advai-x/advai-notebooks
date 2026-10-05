"""Embedding-only corpus retrieval evaluation."""

from advai_notebooks.common.retrieval import retrieve

from .metrics import evaluate_rankings


def evaluate(model, data, cfg, split):
    if cfg.data.reject_truncation:
        model.check_lengths([data.queries[q]["text"] for q in data.query_ids(split)], "query")
        model.check_lengths([d["text"] for d in data.corpus.values()], "document")
    rankings = retrieve(
        model,
        data,
        data.query_ids(split),
        cfg.retrieval.top_k,
        cfg.retrieval.corpus_chunk_size,
        cfg.retrieval.query_chunk_size,
        batch_size=cfg.training.batch_size,
    )
    return evaluate_rankings(rankings, data, split, cfg.evaluation.cutoffs), rankings
