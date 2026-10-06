---
library_name: sentence-transformers
tags:
- advai-notebooks
- embedding
- demo
- synthetic-data
- random-initialization
pipeline_tag: sentence-similarity
---

# widebluesky/wbs-embedding-demo

## Demo release v1

This is a **trained tiny demo model** for testing the training, publication,
download and inference workflow. Its tiny BERT backbone is initialized randomly
and trained for **two epochs** on synthetic fixtures from `advai-notebooks`. Its outputs and metrics do not establish domain quality;
this repository is a placeholder for future domain-trained releases.

## Model and training

```json
{
  "base": "random tiny BERT",
  "hidden_size": 32,
  "embedding_dimension": 32,
  "pooling": "mean",
  "normalized": true,
  "similarity": "dot",
  "loss": "mnrl",
  "learning_rate": 0.001,
  "epochs_completed": 2,
  "seed": 42,
  "optimizer": "adamw",
  "device": "cpu",
  "precision": "fp32",
  "parameter_count": 13408
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
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("widebluesky/wbs-embedding-demo", revision="main", device="cpu")
queries = model.encode_query(["reset password"])
documents = model.encode_document(["account login password reset"])
print(queries.shape, documents.shape)
```

The exported vectors have 32 dimensions and are normalized. The repository also
supports this project's `EmbeddingModel.load(...).encode(..., role="query")` facade.

Pin a commit SHA in `revision` for reproducible inference.

## Synthetic held-out test metrics

These results describe the published weights on synthetic examples only.

```json
{
  "recall@1": 0.5,
  "mrr@1": 1.0,
  "ndcg@1": 1.0,
  "recall@3": 1.0,
  "mrr@3": 1.0,
  "ndcg@3": 0.9885614852711598,
  "recall@5": 1.0,
  "mrr@5": 1.0,
  "ndcg@5": 0.9885614852711598,
  "recall@10": 1.0,
  "mrr@10": 1.0,
  "ndcg@10": 0.9885614852711598
}
```

Only inference weights, tokenizer, model configuration and required runtime
metadata are included. Raw training records, optimizer state, checkpoints for
resuming training and local machine paths are not part of this publication.
