"""Exact cosine retrieval, with bounded corpus/query score blocks."""


def retrieve(
    model,
    data,
    query_ids,
    top_k,
    corpus_chunk_size=1024,
    query_chunk_size=64,
    document_ids=None,
    batch_size=32,
    eligible=None,
):
    import torch

    doc_ids = sorted(data.corpus if document_ids is None else document_ids)
    if not doc_ids:
        raise ValueError("Retrieval corpus is empty")
    query_vectors = model.encode([data.queries[q]["text"] for q in query_ids], "query", batch_size)
    results = {q: [] for q in query_ids}
    for start in range(0, len(doc_ids), corpus_chunk_size):
        block_ids = doc_ids[start : start + corpus_chunk_size]
        document_vectors = model.encode(
            [data.corpus[d]["text"] for d in block_ids], "document", batch_size
        )
        for qstart in range(0, len(query_ids), query_chunk_size):
            scores = query_vectors[qstart : qstart + query_chunk_size] @ document_vectors.T
            if eligible is not None:
                for i, qid in enumerate(query_ids[qstart : qstart + query_chunk_size]):
                    mask = torch.tensor([eligible(qid, doc) for doc in block_ids], dtype=torch.bool)
                    scores[i, ~mask] = -torch.inf
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
            for i, qid in enumerate(query_ids[qstart : qstart + query_chunk_size]):
                merged = results[qid] + [
                    {"doc_id": block_ids[j], "score": float(scores[i, j])}
                    for j in order[i].tolist()
                    if torch.isfinite(scores[i, j])
                ]
                results[qid] = sorted(merged, key=lambda row: (-row["score"], row["doc_id"]))[
                    :top_k
                ]
    return results


def rerank(model, data, candidates, batch_size=32, reject_truncation=True):
    results = {}
    for qid, rows in candidates.items():
        pairs = [(data.queries[qid]["text"], data.corpus[row["doc_id"]]["text"]) for row in rows]
        if reject_truncation:
            model.check_lengths(pairs)
        scores = model.score(pairs, batch_size)
        results[qid] = sorted(
            [
                {"doc_id": row["doc_id"], "score": score}
                for row, score in zip(rows, scores, strict=True)
            ],
            key=lambda row: (-row["score"], row["doc_id"]),
        )
    return results
