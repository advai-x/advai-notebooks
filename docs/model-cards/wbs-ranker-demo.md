---
library_name: sentence-transformers
tags:
- advai-notebooks
- ranker
- demo
- synthetic-data
- random-initialization
pipeline_tag: text-classification
datasets:
- widebluesky/wbs-ranker-demo
---

# widebluesky/wbs-ranker-demo

## Demo release v1

This is a **trained tiny demo model** for testing the training, publication,
download and inference workflow. Its tiny BERT backbone is initialized randomly
and trained for **two epochs** on synthetic fixtures from `advai-notebooks`. Its outputs and metrics do not establish domain quality;
this repository is a placeholder for future domain-trained releases.

## Model and training

```json
{
  "base": "random tiny BERT sequence classifier",
  "hidden_size": 32,
  "num_labels": 1,
  "score_type": "raw_logit",
  "loss": "bce",
  "learning_rate": 0.001,
  "epochs_completed": 2,
  "seed": 42,
  "optimizer": "adamw",
  "device": "cpu",
  "precision": "fp32",
  "parameter_count": 13441
}
```

This release uses the final trained epoch-2 checkpoint. Validation selected
the epoch-0 baseline in the original smoke experiment; that baseline is not
the checkpoint published here. The final epoch is used deliberately for this trained demo.

## Inference

```bash
python -m pip install 'sentence-transformers==5.7.0'
```

```python
import torch
from sentence_transformers import CrossEncoder

model = CrossEncoder("widebluesky/wbs-ranker-demo", revision="main", device="cpu",
                     activation_fn=torch.nn.Identity())
scores = model.predict([("reset password", "account login password reset")],
                       activation_fn=torch.nn.Identity())
print(scores)
```

Scores are raw relevance logits, not probabilities. Higher scores indicate
greater relevance. The repository also supports this project's `RankerModel.load(...).score(...)` facade.

Pin a commit SHA in `revision` for reproducible inference.

## Synthetic held-out test metrics

These results describe the published weights on synthetic examples only.

```json
{
  "recall@1": 0.0,
  "mrr@1": 0.0,
  "ndcg@1": 0.0,
  "recall@3": 0.0,
  "mrr@3": 0.0,
  "ndcg@3": 0.0,
  "recall@5": 0.0,
  "mrr@5": 0.0,
  "ndcg@5": 0.0,
  "recall@10": 1.0,
  "mrr@10": 0.16666666666666666,
  "ndcg@10": 0.3662044332294177
}
```

Only inference weights, tokenizer, model configuration and required runtime
metadata are included. Raw training records, optimizer state, checkpoints for
resuming training and local machine paths are not part of this publication.

## Training and evaluation dataset

Dataset: [widebluesky/wbs-ranker-demo](https://huggingface.co/datasets/widebluesky/wbs-ranker-demo).
Verified dataset revision: [`356d77b9736aa8185480751a9e849d60cfebb6e5`](https://huggingface.co/datasets/widebluesky/wbs-ranker-demo/tree/356d77b9736aa8185480751a9e849d60cfebb6e5) (`demo-v1`).

This independently versioned task dataset contains the exact synthetic records used
for this trained demo release. It includes its own training samples and a complete
local copy of the evaluation benchmark. No other dataset repository is required.
All records were reloaded from the pinned Hub revision and checked against the
original training provenance before associating this dataset with the model.

Training uses `ranker_training` (48 pairs: 24 positive and 24 negative).
Evaluation uses `queries`, `corpus`, `qrels` and the explicit `candidates` configuration.
Embedding contrastive samples are not included; no Embedding model is required.

```python
from datasets import load_dataset

data = load_dataset(
    "widebluesky/wbs-ranker-demo", "ranker_training",
    revision="356d77b9736aa8185480751a9e849d60cfebb6e5", token=True,
)
```

Access uses the existing Hugging Face login; both model and dataset repositories are private.
Original project-native data files are available under `source/` in the dataset repository.
Canonical benchmark fingerprint: `f96589e88e79ce684f3c8733ae1a3c98a8f028459f6e82e3477efeb2867b2a19`.
Training sample fingerprint: `4559aec5e94271f5af8da60eaacf855a3b3033c7130272f0a68709cbf78653fc`.

This association supersedes the former shared retrieval dataset association.
The original `widebluesky/wbs-retrieval-demo` dataset and its pinned `demo-v1` revision
remain available for historical reproducibility.
