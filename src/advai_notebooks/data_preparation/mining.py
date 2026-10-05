"""Training-only judgments or an explicitly selected offline retriever."""

from advai_notebooks.common.retrieval import retrieve


def mine_negatives(model, data, cfg):
    qids, doc_ids = data.query_ids("train"), data.training_doc_ids()
    positive_texts = {
        qid: {data.corpus[d]["text"].strip() for d in data.positives(qid)} for qid in qids
    }

    def eligible(qid, doc):
        return data.corpus[doc]["text"].strip() not in positive_texts[qid] and (
            data.qrels[qid].get(doc) == 0
            or (cfg.data.allow_unjudged_negatives and doc not in data.qrels[qid])
        )

    # Mine from training judgments only. Held-out labels never select negatives.
    if cfg.mining.strategy == "judged":
        rankings = {
            q: [{"doc_id": d, "score": 0.0} for d in doc_ids if eligible(q, d)][
                : cfg.mining.negatives_per_query
            ]
            for q in qids
        }
    else:
        rankings = retrieve(
            model,
            data,
            qids,
            cfg.mining.negatives_per_query,
            cfg.retrieval.corpus_chunk_size,
            cfg.retrieval.query_chunk_size,
            doc_ids,
            cfg.training.batch_size,
            eligible=eligible,
        )
    rows = []
    for qid in qids:
        positive = data.positives(qid)
        negatives = [r["doc_id"] for r in rankings[qid]]
        count = cfg.mining.negatives_per_query
        if len(negatives) < count:
            raise ValueError(
                f"Query {qid} has only {len(negatives)} eligible negatives; need {count}"
            )
        for doc in sorted(positive):
            rows.append({"query_id": qid, "positive_id": doc, "negative_ids": negatives[:count]})
    return rows


def ranker_rows(mined):
    rows = {}
    for row in mined:
        qid = row["query_id"]
        rows[qid, row["positive_id"]] = 1.0
        for doc in row["negative_ids"]:
            rows[qid, doc] = 0.0
    if not rows or not {0.0, 1.0} <= set(rows.values()):
        raise ValueError("Ranker training requires both positive and negative pairs")
    return [
        {"query_id": qid, "doc_id": doc, "label": label}
        for (qid, doc), label in sorted(rows.items())
    ]
