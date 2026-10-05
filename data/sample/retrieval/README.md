# Shared retrieval data contract

Three UTF-8 JSONL files describe one retrieval benchmark. Each row is one object.

```json
{"doc_id":"doc-1","text":"reset your account password","group_id":"source-1"}
{"query_id":"query-1","text":"how to reset password","group_id":"query-source-1","split":"train"}
{"query_id":"query-1","doc_id":"doc-1","relevance":3}
```

These examples belong to `corpus.jsonl`, `queries.jsonl` and `qrels.jsonl`
respectively. IDs, groups and text must be nonempty strings. Documents and queries
may carry optional `metadata`; other fields are rejected. Query splits are
`train`, `validation`, `test`, and all three must be populated for training.
Every query needs at least one judged positive. IDs must be unique and every
qrel reference must exist. Duplicate query/document judgments are errors.

Relevance is a finite number in `[0, 3]`: zero means explicitly nonrelevant,
positive means relevant. Embedding trains with every positive; ranker trains a
binary relevance objective. Evaluation nDCG preserves the graded judgments.
The absence of a qrel means unjudged, not a negative. Default training requires
explicit negative judgments. Keep qrels coverage sufficient for mining.

Query groups and identical query text cannot cross splits. A group should
represent the upstream conversation, document, user or other leakage unit;
do not invent independent groups for paraphrases of the same source.
Optional document-group isolation prevents positive source overlap and holds
validation/test positive groups out of mining. It also rejects identical
positive document text crossing splits. This is separate from shared-corpus,
new-query evaluation, where document reuse is intentional.

The checked-in fixture has 12 documents, 18 queries (12 train, 3 validation,
3 test), and exhaustive synthetic judgments. Query templates and vocabulary
overlap deliberately to make a tiny offline test executable. These examples
are not an independent benchmark and cannot establish generalization quality.
Use realistic held-out sources, time windows and independent queries for domain
quality assessment. Put those files under ignored `data/local/retrieval/`.

## Independent prepared files

`embedding_samples.json` and `ranker_samples.json` have `schema_version: 1`,
`kind: "embedding" | "ranker"`, `data_fingerprint`, optional provenance `source`,
and a `rows` list. Each file is selected by its own training config.

```json
{"query_id":"query-1","positive_id":"doc-1","negative_ids":["doc-2"]}
{"query_id":"query-1","doc_id":"doc-1","label":1}
```

The first row is embedding-only, the second ranker-only. Plain JSONL of those
rows is also accepted and revalidated against this experiment's corpus/qrels.
All training queries must be represented. Embedding requires equal explicit
negative counts; ranker requires both binary classes. No held-out queries or
known-positive negative labels are accepted.

An external `candidates.json` follows this shape:

```json
{"schema_version":1,"data_fingerprint":"<canonical corpus/query/qrels hash>","source":{"type":"external retriever"},"splits":{"validation":{"query-v":[{"doc_id":"doc-2","score":0.8}]},"test":{"query-t":[{"doc_id":"doc-3","score":0.7}]}}}
```

Generate the canonical fingerprint from `load_data(cfg).fingerprint`. Candidate
membership is independent of relevance labels, and missing positives must not
be added for metric inflation. The checked-in candidate fixture contains the
whole tiny corpus for each held-out query with neutral original scores. It is a
ranker-only full-corpus smoke benchmark, not a trained retriever's output.

Preparation outputs are immutable per-run files. `advai-data prepare` can
produce these contracts separately; each trainer only reads and validates them.
