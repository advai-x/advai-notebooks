"""File contracts shared by preparation tools and independent trainers."""

import json
import math
from pathlib import Path

from .artifacts import digest
from .retrieval_data import read_jsonl


def load_samples(path, data, kind):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema_version") != 1
            or artifact.get("kind") != kind
            or artifact.get("data_fingerprint") != data.fingerprint
        ):
            raise ValueError("Training artifact kind/version/dataset fingerprint mismatch")
        rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Training samples must be a nonempty list of objects")
    return rows, digest(rows)


def training_reference(data, qid, doc):
    if (
        not isinstance(qid, str)
        or not isinstance(doc, str)
        or qid not in data.queries
        or doc not in data.corpus
    ):
        raise ValueError("Unknown training query/document")
    if data.queries[qid]["split"] != "train":
        raise ValueError("Training samples may only reference train queries")
    if data.enforce_document_groups:
        heldout = {
            data.corpus[d]["group_id"]
            for q, row in data.queries.items()
            if row["split"] != "train"
            for d in data.positives(q)
        }
        if data.corpus[doc]["group_id"] in heldout:
            raise ValueError("Training document belongs to a held-out source group")


def validate_negative(data, qid, doc, allow_unjudged):
    training_reference(data, qid, doc)
    positive_texts = {data.corpus[d]["text"].strip() for d in data.positives(qid)}
    if data.corpus[doc]["text"].strip() in positive_texts:
        raise ValueError("Known positive or identical positive text used as a negative")
    label = data.qrels[qid].get(doc)
    if label != 0 and not (label is None and allow_unjudged):
        raise ValueError("Unjudged negative requires data.allow_unjudged_negatives")


def load_candidates(path, data, splits=("validation", "test")):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("data_fingerprint") != data.fingerprint:
        raise ValueError("Candidates and dataset fingerprints differ")
    if artifact.get("schema_version", 1) != 1 or not isinstance(artifact.get("splits"), dict):
        raise ValueError("Invalid candidate artifact")
    for split in splits:
        candidates = artifact["splits"].get(split)
        if not isinstance(candidates, dict):
            raise ValueError(f"Missing {split} candidates")
        if set(candidates) != set(data.query_ids(split)):
            raise ValueError("Candidates must cover exactly the requested split")
        for rows in candidates.values():
            if not isinstance(rows, list) or not rows:
                raise ValueError("Each query must have a nonempty candidate list")
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"doc_id", "score"}:
                    raise ValueError("Candidates require doc_id and score")
                if (
                    not isinstance(row["doc_id"], str)
                    or isinstance(row["score"], bool)
                    or not isinstance(row["score"], (float, int))
                    or not math.isfinite(row["score"])
                ):
                    raise ValueError("Invalid candidate ID/score")
            docs = [row["doc_id"] for row in rows]
            if len(docs) != len(set(docs)) or any(doc not in data.corpus for doc in docs):
                raise ValueError("Candidates contain duplicate or unknown documents")
    return {split: artifact["splits"][split] for split in splits}
