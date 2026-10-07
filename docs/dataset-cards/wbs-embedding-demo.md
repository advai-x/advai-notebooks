---
pretty_name: Synthetic embedding demo fixtures
language:
- en
size_categories:
- n<1K
tags:
- advai-notebooks
- embedding
- demo
- synthetic-data
task_categories:
- text-retrieval
configs:
- config_name: queries
  data_files:
  - split: train
    path: data/queries/train.jsonl
  - split: validation
    path: data/queries/validation.jsonl
  - split: test
    path: data/queries/test.jsonl
- config_name: corpus
  data_files:
  - split: corpus
    path: data/corpus/corpus.jsonl
- config_name: qrels
  data_files:
  - split: train
    path: data/qrels/train.jsonl
  - split: validation
    path: data/qrels/validation.jsonl
  - split: test
    path: data/qrels/test.jsonl
- config_name: embedding_training
  data_files:
  - split: train
    path: data/embedding_training/train.jsonl
  default: true
---

# widebluesky/wbs-embedding-demo

## Independent task dataset v1

This private dataset packages the synthetic English fixtures used by the trained
embedding demo model. It includes its own training records and a self-contained copy
of the corpus, queries and relevance judgments needed for evaluation.
Loading and training do not require another dataset repository or model task.
Original project-native files are preserved byte-for-byte under `source/`;
`data/` contains explicit Datasets JSONL loading views with unchanged records.

These fixtures were authored in `advai-notebooks`, with no customer data or external
benchmark source. They verify engineering contracts, not domain quality.

## Configurations and split counts

| Configuration | Split counts |
| --- | --- |
| `queries` | train: 12, validation: 3, test: 3 |
| `corpus` | corpus: 12 |
| `qrels` | train: 144, validation: 36, test: 36 |
| `embedding_training` | train: 24 |

The default configuration is `embedding_training`.

## Schema and evaluation semantics

- `queries`: `query_id`, `text`, `group_id`, `split`.
- `corpus`: `doc_id`, `text`, `group_id`; the corpus is shared across query evaluation splits within this dataset.
- `qrels`: `query_id`, `doc_id`, `relevance` (0 to 3), assigned to the referenced query split.
- `embedding_training`: `query_id`, `positive_id`, `negative_ids`; 24 contrastive training examples.

There are 12 documents, 18 queries and 216 synthetic judgments. Training views contain
only train queries; validation/test query IDs remain in the separate benchmark views.
Query groups and identical queries do not cross splits. Positive document source groups
are isolated by query split. Negative labels are checked against judgments.

Ranker training pairs and candidates are excluded from this dataset.

## Loading

```python
from datasets import load_dataset

samples = load_dataset("widebluesky/wbs-embedding-demo", "embedding_training", revision="demo-v1", token=True)
queries = load_dataset("widebluesky/wbs-embedding-demo", "queries", revision="demo-v1", token=True)
```

Use an existing Hugging Face login; do not embed credentials. Pin the dataset commit
SHA for reproducibility. For project-native training, download a dataset snapshot and
point task configuration paths at `source/corpus.jsonl`, `source/queries.jsonl`,
`source/qrels.jsonl` and `source/embedding_samples.json`.

## Trained model and provenance

Corresponding model: [widebluesky/wbs-embedding-demo](https://huggingface.co/widebluesky/wbs-embedding-demo).

Canonical benchmark fingerprint: `f96589e88e79ce684f3c8733ae1a3c98a8f028459f6e82e3477efeb2867b2a19`.

Training sample fingerprint: `cebc151f0e9c04eae8c92a1d79a8c3ae6771b1a81fb7484e7a13a34555ee9173`.

Project validators verify formats, IDs, references, splits and training/candidate rules.
Fingerprints match the saved records from the run that produced the published model.
Every configuration is reloaded locally and from a pinned Hub commit and all records
are compared before updating model associations. Source hashes, view fingerprints
and published model weight hashes are recorded in `dataset_manifest.json`.
No weights, optimizer state, logs, credentials or machine-specific paths are included.

## Migration

This task dataset supersedes the shared repository for this model's active association.
The original shared dataset remains available at its historical revision:
[widebluesky/wbs-retrieval-demo](https://huggingface.co/datasets/widebluesky/wbs-retrieval-demo/tree/c75ca4e872c63c4683b266a4924a493968a1111f).
The same small benchmark records are intentionally copied into each task dataset;
training records and dataset versioning are maintained independently.
