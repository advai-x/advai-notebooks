"""Validate labeled pairs and external candidates without a retriever dependency."""

from advai_notebooks.common.prepared_data import (
    load_candidates,
    load_samples,
    training_reference,
    validate_negative,
)


def load_training_samples(cfg, data):
    rows, fingerprint = load_samples(cfg.path(cfg.data.training_samples), data, "ranker")
    seen = set()
    for row in rows:
        if set(row) != {"query_id", "doc_id", "label"}:
            raise ValueError("Ranker samples require query_id, doc_id, label")
        qid, doc, label = row["query_id"], row["doc_id"], row["label"]
        training_reference(data, qid, doc)
        if isinstance(label, bool) or not isinstance(label, (float, int)) or label not in (0, 1):
            raise ValueError("Ranker label must be zero or one")
        if (qid, doc) in seen:
            raise ValueError("Duplicate ranker training pair")
        seen.add((qid, doc))
        if label == 1 and doc not in data.positives(qid):
            raise ValueError("Ranker positive must be a positive qrel")
        if label == 0:
            validate_negative(data, qid, doc, cfg.data.allow_unjudged_negatives)
    if {r["query_id"] for r in rows} != set(data.query_ids("train")):
        raise ValueError("Ranker samples must cover all train queries")
    if {r["label"] for r in rows} != {0, 1}:
        raise ValueError("Ranker training requires positive and negative pairs")
    return rows, fingerprint


def load_evaluation_candidates(cfg, data, splits=("validation", "test")):
    return load_candidates(cfg.path(cfg.data.candidates), data, splits)
