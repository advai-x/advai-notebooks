---
pretty_name: Synthetic retrieval demo fixtures
language:
- en
size_categories:
- n<1K
tags:
- advai-notebooks
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
  default: true
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
- config_name: ranker_training
  data_files:
  - split: train
    path: data/ranker_training/train.jsonl
- config_name: candidates
  data_files:
  - split: validation
    path: data/candidates/validation.jsonl
  - split: test
    path: data/candidates/test.jsonl
---

# widebluesky/wbs-retrieval-demo

## Historical combined dataset

The active demo model associations now use independent task datasets:

- [Embedding dataset](https://huggingface.co/datasets/widebluesky/wbs-embedding-demo).
- [Ranker dataset](https://huggingface.co/datasets/widebluesky/wbs-ranker-demo).

This combined dataset is retained for historical reproducibility. Its original
`demo-v1` tag still pins `c75ca4e872c63c4683b266a4924a493968a1111f`.
The training records and benchmark files remain available at that revision.

## Demo dataset v1

Small, deliberately synthetic English fixtures used to train and evaluate the
corresponding tiny demo models from `advai-notebooks`. The records were authored
as repository fixtures, with no external benchmark or customer data source.
This dataset verifies engineering paths and data contracts; it does not establish
domain quality or independent generalization performance.

Original source files are preserved byte-for-byte under `source/`. JSONL views under
`data/` expose explicit subsets and splits through the Hugging Face Datasets library.
The views preserve original values and ordering within each split.

## Configurations and splits

| Configuration | Split counts |
| --- | --- |
| queries | train: 12, validation: 3, test: 3 |
| corpus | corpus: 12 |
| qrels | train: 144, validation: 36, test: 36 |
| embedding_training | train: 24 |
| ranker_training | train: 48 |
| candidates | validation: 36, test: 36 |

## Schema and split semantics

- `queries`: `query_id`, `text`, `group_id`, `split`.
- `corpus`: `doc_id`, `text`, `group_id`; the `corpus` split is shared across query evaluation splits.
- `qrels`: `query_id`, `doc_id`, `relevance` (0 to 3), grouped by the referenced query split.
- `embedding_training`: `query_id`, `positive_id`, `negative_ids`.
- `ranker_training`: `query_id`, `doc_id`, binary `label` (24 positive and 24 negative pairs).
- `candidates`: `query_id`, `doc_id`, `score`, flattened without changing candidate membership.

There are 12 documents, 18 queries and 216 exhaustive synthetic judgments.
Query groups and identical queries do not cross splits. Positive document source groups
are isolated across splits. Training samples reference only training queries and allowed
training document groups. Unjudged documents are not implicitly treated as negatives.
Candidate lists contain the full tiny corpus and neutral scores; they are not the output
of a trained retriever. Repeated templates and vocabulary make this a smoke fixture.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("widebluesky/wbs-retrieval-demo", "queries", revision="demo-v1", token=True)
print(dataset)
```

These repositories are private. Use your existing Hugging Face login for access;
do not embed credentials in code or dataset files. Pin the dataset commit SHA for reproducibility.

For project-native file loading, download the dataset snapshot and point configuration
paths at its original `source/` files. The derived subsets are additional loading views.

## Corresponding trained models

- [widebluesky/wbs-embedding-demo](https://huggingface.co/widebluesky/wbs-embedding-demo): verified training data fingerprint `f96589e88e79ce684f3c8733ae1a3c98a8f028459f6e82e3477efeb2867b2a19`.
- [widebluesky/wbs-ranker-demo](https://huggingface.co/widebluesky/wbs-ranker-demo): verified training data fingerprint `f96589e88e79ce684f3c8733ae1a3c98a8f028459f6e82e3477efeb2867b2a19`.

## Validation and provenance

Project validators checked IDs, schemas, references, labels, split groups and negative/candidate rules.
Dataset fingerprints match saved records from the runs that produced the published model weights.
The publication process reloads every subset locally and from a pinned Hub commit and compares all records.
Source file hashes, subset record fingerprints and model weight hashes are recorded in `dataset_manifest.json`.
Model weights, training state, logs and machine-specific paths are not included.

Source implementation commit: `f43a44e2431bae9d0451b445cc3bde5f1e2b8493`.
