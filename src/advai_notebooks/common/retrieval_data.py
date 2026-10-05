"""Canonical corpus/queries/qrels and query-group split checks."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .artifacts import digest

SPLITS = {"train", "validation", "test"}


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Each line must be an object")
            rows.append(value)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


@dataclass
class RetrievalData:
    corpus: dict
    queries: dict
    qrels: dict
    enforce_document_groups: bool
    fingerprint: str

    def query_ids(self, split):
        ids = sorted(qid for qid, row in self.queries.items() if row["split"] == split)
        if not ids:
            raise ValueError(f"No queries in {split} split")
        return ids

    def positives(self, qid):
        return {doc for doc, relevance in self.qrels[qid].items() if relevance > 0}

    def training_doc_ids(self):
        train_ids = {d for qid in self.query_ids("train") for d in self.qrels[qid]}
        if self.enforce_document_groups:
            heldout_groups = {
                self.corpus[d]["group_id"]
                for q, row in self.queries.items()
                if row["split"] != "train"
                for d in self.positives(q)
            }
            train_ids = {d for d in train_ids if self.corpus[d]["group_id"] not in heldout_groups}
        return sorted(train_ids)

    def summary(self):
        return {
            "documents": len(self.corpus),
            "queries": len(self.queries),
            "judgments": sum(len(v) for v in self.qrels.values()),
            "queries_by_split": dict(Counter(q["split"] for q in self.queries.values())),
            "fingerprint": self.fingerprint,
            "document_group_isolation": self.enforce_document_groups,
        }


def load_data(cfg):
    documents = read_jsonl(cfg.path(cfg.data.corpus))
    queries = read_jsonl(cfg.path(cfg.data.queries))
    judgments = read_jsonl(cfg.path(cfg.data.qrels))
    corpus, query_map = {}, {}
    for rows, table, id_key in ((documents, corpus, "doc_id"), (queries, query_map, "query_id")):
        required = {id_key, "text", "group_id"} | ({"split"} if id_key == "query_id" else set())
        for row in rows:
            if not required <= set(row) or set(row) - (required | {"metadata"}):
                raise ValueError(f"Unexpected fields in {id_key} row")
            for key in (id_key, "text", "group_id"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"{key} must be nonempty text")
            if row[id_key] in table:
                raise ValueError(f"Duplicate {id_key}: {row[id_key]}")
            table[row[id_key]] = row
    group_splits, text_splits = {}, {}
    for q in queries:
        if not isinstance(q["split"], str) or q["split"] not in SPLITS:
            raise ValueError("Query split must be train, validation or test")
        for table, key in ((group_splits, q["group_id"]), (text_splits, q["text"].strip())):
            if key in table and table[key] != q["split"]:
                raise ValueError("Query group or identical query crosses splits")
            table[key] = q["split"]
    qrels = {qid: {} for qid in query_map}
    for row in judgments:
        if set(row) != {"query_id", "doc_id", "relevance"}:
            raise ValueError("Qrels require query_id, doc_id and relevance")
        qid, doc, rel = row["query_id"], row["doc_id"], row["relevance"]
        if (
            not isinstance(qid, str)
            or not isinstance(doc, str)
            or qid not in query_map
            or doc not in corpus
        ):
            raise ValueError("Qrel references an unknown query/document")
        if (
            isinstance(rel, bool)
            or not isinstance(rel, (int, float))
            or not math.isfinite(rel)
            or not 0 <= rel <= 3
        ):
            raise ValueError("Relevance must be finite in [0, 3]")
        if doc in qrels[qid]:
            raise ValueError("Duplicate query/document judgment")
        qrels[qid][doc] = float(rel)
    if any(not any(rel > 0 for rel in labels.values()) for labels in qrels.values()):
        raise ValueError("Every query requires at least one positive judgment")
    if cfg.data.enforce_document_groups:
        document_splits = {}
        document_text_splits = {}
        for qid, labels in qrels.items():
            for doc, rel in labels.items():
                if rel <= 0:
                    continue
                group = corpus[doc]["group_id"]
                split = query_map[qid]["split"]
                if group in document_splits and document_splits[group] != split:
                    raise ValueError("Positive document source group crosses splits")
                document_splits[group] = split
                text = corpus[doc]["text"].strip()
                if text in document_text_splits and document_text_splits[text] != split:
                    raise ValueError("Identical positive document text crosses splits")
                document_text_splits[text] = split
    return RetrievalData(
        corpus,
        query_map,
        qrels,
        cfg.data.enforce_document_groups,
        digest({"corpus": documents, "queries": queries, "qrels": judgments}),
    )
