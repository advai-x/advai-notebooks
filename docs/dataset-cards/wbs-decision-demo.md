---
pretty_name: Synthetic Decision triage demo fixtures
language:
- en
size_categories:
- n<1K
tags:
- advai-notebooks
- demo
- synthetic-data
task_categories:
- text-classification
configs:
- config_name: triage
  data_files:
  - split: train
    path: data/triage/train.jsonl
  - split: validation
    path: data/triage/validation.jsonl
  - split: calibration
    path: data/triage/calibration.jsonl
  - split: test
    path: data/triage/test.jsonl
  default: true
---

# widebluesky/wbs-decision-demo

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
| triage | train: 12, validation: 4, calibration: 10, test: 4 |

## Schema and split semantics

Each record contains `id`, `group_id`, `split`, `task`, `state` and `gold`.
`source/tasks.json` supplies questions, option definitions and output types.
The `triage` task has department (`choice`), urgency (`score`) and churn risk (`noul`).
Each of the 30 message records has three labels, for 90 individual decisions.
The 10 calibration records are separate from training, validation and test and are
used only to fit temperature scaling. Groups and identical states do not cross splits.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset("widebluesky/wbs-decision-demo", "triage", revision="demo-v1", token=True)
print(dataset)
```

These repositories are private. Use your existing Hugging Face login for access;
do not embed credentials in code or dataset files. Pin the dataset commit SHA for reproducibility.

For project-native file loading, download the dataset snapshot and point configuration
paths at its original `source/` files. The derived subsets are additional loading views.

## Corresponding trained models

- [widebluesky/wbs-decision-demo](https://huggingface.co/widebluesky/wbs-decision-demo): verified training data fingerprint `2ed63814602e43b30c66678574f4808d06b3fd01e829eab4c06f039d16be4c11`.

## Validation and provenance

Project validators checked IDs, schemas, references, labels, split groups and negative/candidate rules.
Dataset fingerprints match saved records from the runs that produced the published model weights.
The publication process reloads every subset locally and from a pinned Hub commit and compares all records.
Source file hashes, subset record fingerprints and model weight hashes are recorded in `dataset_manifest.json`.
Model weights, training state, logs and machine-specific paths are not included.

Source implementation commit: `f43a44e2431bae9d0451b445cc3bde5f1e2b8493`.
