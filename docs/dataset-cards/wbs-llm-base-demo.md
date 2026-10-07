---
pretty_name: Synthetic LLM Base CPT demo fixtures
language:
- en
size_categories:
- n<1K
tags:
- advai-notebooks
- demo
- synthetic-data
task_categories:
- text-generation
configs:
- config_name: base
  data_files:
  - split: train
    path: data/base/train.jsonl
  - split: validation
    path: data/base/validation.jsonl
  - split: test
    path: data/base/test.jsonl
  default: true
---

# widebluesky/wbs-llm-base-demo

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
| base | train: 8, validation: 2, test: 2 |

## Schema and split semantics

Each record contains `id`, `document_id`, `text` and `split`.
The content includes small general-fact and account-support statements.
Training supervises next tokens in document text; padding is masked.
Document groups and normalized duplicate content do not cross splits.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("widebluesky/wbs-llm-base-demo", "base", revision="demo-v1", token=True)
print(dataset)
```

These repositories are private. Use your existing Hugging Face login for access;
do not embed credentials in code or dataset files. Pin the dataset commit SHA for reproducibility.

For project-native file loading, download the dataset snapshot and point configuration
paths at its original `source/` files. The derived subsets are additional loading views.

## Corresponding trained models

- [widebluesky/wbs-llm-base-demo](https://huggingface.co/widebluesky/wbs-llm-base-demo): verified training data fingerprint `26aec5cfd5b21082af1e8f772dfe41f447ca0275f5cc6b8b8b1d98334e791335`.

## Validation and provenance

Project validators checked IDs, schemas, references, labels, split groups and negative/candidate rules.
Dataset fingerprints match saved records from the runs that produced the published model weights.
The publication process reloads every subset locally and from a pinned Hub commit and compares all records.
Source file hashes, subset record fingerprints and model weight hashes are recorded in `dataset_manifest.json`.
Model weights, training state, logs and machine-specific paths are not included.

Source implementation commit: `f43a44e2431bae9d0451b445cc3bde5f1e2b8493`.
