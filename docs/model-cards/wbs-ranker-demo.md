---
library_name: sentence-transformers
tags:
- advai-notebooks
- ranker
- demo
- synthetic-data
- random-initialization
pipeline_tag: text-classification
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
