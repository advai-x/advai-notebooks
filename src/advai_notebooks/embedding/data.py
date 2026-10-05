"""Validate prepared contrastive samples independently of negative mining."""

from advai_notebooks.common.prepared_data import load_samples, training_reference, validate_negative


def load_training_samples(cfg, data):
    rows, fingerprint = load_samples(cfg.path(cfg.data.training_samples), data, "embedding")
    seen, negative_count = set(), None
    for row in rows:
        if set(row) != {"query_id", "positive_id", "negative_ids"}:
            raise ValueError("Embedding samples require query_id, positive_id, negative_ids")
        qid, doc, negatives = row["query_id"], row["positive_id"], row["negative_ids"]
        training_reference(data, qid, doc)
        if doc not in data.positives(qid):
            raise ValueError("Embedding positive must be a positive qrel")
        if (qid, doc) in seen:
            raise ValueError("Duplicate embedding training pair")
        seen.add((qid, doc))
        if (
            not isinstance(negatives, list)
            or any(not isinstance(d, str) for d in negatives)
            or len(set(negatives)) != len(negatives)
        ):
            raise ValueError("negative_ids must be distinct strings")
        if negative_count is not None and len(negatives) != negative_count:
            raise ValueError("Embedding samples must have equal explicit negative counts")
        negative_count = len(negatives)
        for negative in negatives:
            validate_negative(data, qid, negative, cfg.data.allow_unjudged_negatives)
        texts = [data.corpus[d]["text"].strip() for d in [doc, *negatives]]
        if len(texts) != len(set(texts)):
            raise ValueError("Duplicate candidate text in embedding sample")
    if {r["query_id"] for r in rows} != set(data.query_ids("train")):
        raise ValueError("Embedding samples must cover all train queries")
    return rows, fingerprint
