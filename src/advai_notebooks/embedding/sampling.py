"""Embedding-only batching without known in-batch false negatives."""

import random


def embedding_batches(rows, data, batch_size, seed):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    batches = []
    for row in ordered:
        qid = row["query_id"]
        docs = [row["positive_id"], *row["negative_ids"]]
        doc_texts = {data.corpus[d]["text"].strip() for d in docs}
        own_positive_texts = {data.corpus[d]["text"].strip() for d in data.positives(qid)}
        if len(doc_texts) != len(docs):
            raise ValueError("Embedding candidates contain duplicate document text")
        for batch in batches:
            if len(batch) >= batch_size:
                continue
            compatible = True
            for other in batch:
                other_qid = other["query_id"]
                other_docs = [other["positive_id"], *other["negative_ids"]]
                other_texts = {data.corpus[d]["text"].strip() for d in other_docs}
                other_positive = {data.corpus[d]["text"].strip() for d in data.positives(other_qid)}
                if (
                    data.queries[qid]["group_id"] == data.queries[other_qid]["group_id"]
                    or doc_texts & other_texts
                    or doc_texts & other_positive
                    or other_texts & own_positive_texts
                ):
                    compatible = False
                    break
            if compatible:
                batch.append(row)
                break
        else:
            batches.append([row])
    if any(len(batch) == 1 and not batch[0]["negative_ids"] for batch in batches):
        raise ValueError(
            "MNRL needs negative candidates; increase negatives_per_query or batch_size/data diversity"
        )
    return batches
