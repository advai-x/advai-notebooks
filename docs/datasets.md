# Demo models and Dataset associations

English | [简体中文](datasets.zh-CN.md)

The recorded seven model releases each have a corresponding private Hugging Face
Dataset with English model/data cards. Embedding and Ranker maintain separate
Dataset repositories, training samples and versions. Each includes a complete
evaluation benchmark copy and loads independently; the current demo benchmarks
have identical content. This page describes recorded releases, not a fresh audit
of current Hub repository visibility or branch contents.

| Model | Dataset | Training configuration |
| --- | --- | --- |
| `widebluesky/wbs-embedding-demo` | [widebluesky/wbs-embedding-demo](https://huggingface.co/datasets/widebluesky/wbs-embedding-demo) | `embedding_training` |
| `widebluesky/wbs-ranker-demo` | [widebluesky/wbs-ranker-demo](https://huggingface.co/datasets/widebluesky/wbs-ranker-demo) | `ranker_training` |
| `widebluesky/wbs-decision-demo` | [widebluesky/wbs-decision-demo](https://huggingface.co/datasets/widebluesky/wbs-decision-demo) | `triage` |
| `widebluesky/wbs-llm-base-demo` | [widebluesky/wbs-llm-base-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-base-demo) | `base` |
| `widebluesky/wbs-llm-instruct-demo` | [widebluesky/wbs-llm-instruct-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-instruct-demo) | `instruct` |
| `widebluesky/wbs-llm-ppo-demo` | [widebluesky/wbs-llm-ppo-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-ppo-demo) | `ppo` |
| `widebluesky/wbs-llm-grpo-demo` | [widebluesky/wbs-llm-grpo-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-grpo-demo) | `grpo` |

## Recorded data versions

Each Dataset's recorded `demo-v1` tag pins the following verified commit.
Model-card bodies also record these SHAs.

| Dataset | Pinned commit |
| --- | --- |
| `widebluesky/wbs-embedding-demo` | `d5d422450670f840377c9f8bb3b11b7919cdadfa` |
| `widebluesky/wbs-ranker-demo` | `356d77b9736aa8185480751a9e849d60cfebb6e5` |
| `widebluesky/wbs-decision-demo` | `ccdf91daf6b2810dbf63eda37e69dd2ab3af8f6c` |
| `widebluesky/wbs-llm-base-demo` | `de84c42b70656ff5875602b1abf829c8b2c962f9` |
| `widebluesky/wbs-llm-instruct-demo` | `21213b40998690495308dac1dae09d669846579b` |
| `widebluesky/wbs-llm-ppo-demo` | `5b621b1b02f8f097a2036c0aa0f915e4df5b8e31` |
| `widebluesky/wbs-llm-grpo-demo` | `3246400626f9c419fd1eaa4b36a1594b53089be2` |

Recorded model-main cards use standard `datasets` metadata to link these repositories,
with pinned revisions and loading examples. Dataset-association updates for the
first five models changed only cards/README verification records; their original
model `demo-v1` tags remained at the original commits. PPO/GRPO releases contain
trained full models and independent Datasets, with pinned Dataset SHAs in the cards.
Models and Datasets may share a repo ID but have different `repo_type` values;
Dataset URLs must include `/datasets/`.

## Loading and project training inputs

Install the verified loader version in a separate data-inspection environment:

```bash
python -m pip install 'datasets==4.8.5'
```

```python
from datasets import load_dataset

data = load_dataset(
    "widebluesky/wbs-llm-instruct-demo", "instruct",
    revision="21213b40998690495308dac1dae09d669846579b", token=True,
)
print(data)
```

`token=True` uses an existing HF login; never embed tokens in code. Access to
private repositories is required. Recorded validation used `load_dataset` directly
against pinned Hub SHAs, not just local files.

Each Dataset retains original files in `source/`; `data/` holds split-specific JSONL
views. To use the project trainers, download the Dataset snapshot and point task
configs at the original `source/` files:

- Embedding: `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl`, `embedding_samples.json`.
- Ranker: `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl`, `ranker_samples.json`, `candidates.json`.
- Decision: `triage.jsonl`, `tasks.json`.
- Base/Instruct/PPO/GRPO: `base.jsonl`, `instruct.jsonl`, `ppo.jsonl`, `grpo.jsonl`, respectively.

Embedding has four configurations: `queries`, `corpus`, `qrels` and
`embedding_training` (default, 24 train rows). Ranker has five: `queries`, `corpus`,
`qrels`, `ranker_training` (default, 48 train rows) and `candidates`.
Queries have 12 train/3 validation/3 test rows; corpus has 12 documents and qrels
216 judgments. Training configurations contain only train; evaluation splits
live in separate benchmark configurations. Decision `triage` retains an independent
calibration split. Base/Instruct have train/validation/test. PPO/GRPO each have one
default config with 4 train/2 validation/2 test rows; references serve rewards/evaluation only.

## Recorded validation scope

- Formats, unique IDs, group/content boundaries, references and positive/negative label rules.
- Dataset fingerprints matching the training records that produced released weights.
- Seven Datasets with 14 configurations. The two PPO/GRPO configurations were reloaded row by row locally and from the Hub, including default selection; the earlier five retain their verified pinned versions.
- Independent task packages containing only that task's training samples and passing project validation and standard `load_dataset` loading.
- Matching original bytes and published SHA256 values.
- Matching card metadata, pinned revisions and README inventories, with other model-file hashes unchanged during card updates.

All data is English synthetic smoke fixtures: it verifies mechanics/provenance,
not domain quality or generalization. Complete provenance/config fingerprints
are in each `dataset_manifest.json`; card sources are in [dataset-cards](dataset-cards/).
For a real release, validate and publish the Dataset before linking its ID and
commit from the model card. Demo cards describe only their recorded data/weights;
do not reuse their claims for other training corpora.

## Migration from the shared retrieval Dataset

The historical combined `widebluesky/wbs-retrieval-demo` Dataset retains its
recorded `demo-v1` commit `c75ca4e872c63c4683b266a4924a493968a1111f`.
The migrated Embedding/Ranker model-main cards link only their individual Datasets.
Old Dataset/model tags preserve historical reproducibility; future task Datasets
can evolve independently. Migration split Dataset packages and updated card links
without changing model weights.

The [RL release record](releases/20261008-rl-demo-v1.json) lists private PPO/GRPO
`demo-v1` model/Dataset SHAs and Hub-download hash, inference and test-report checks.
Both models have test exact match and mean reward of 0.0; they demonstrate an
end-to-end engineering workflow only.
