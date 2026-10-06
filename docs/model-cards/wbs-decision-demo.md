---
library_name: pytorch
tags:
- advai-notebooks
- decision
- demo
- synthetic-data
- random-initialization
---

# widebluesky/wbs-decision-demo

## Demo release v1

This is a **trained tiny demo model** for testing the training, publication,
download and inference workflow. Its tiny BERT backbone is initialized randomly
and trained for **two epochs** on synthetic fixtures from `advai-notebooks`. Its outputs and metrics do not establish domain quality;
this repository is a placeholder for future domain-trained releases.

## Model and training

```json
{
  "base": "random tiny BERT",
  "hidden_size": 64,
  "head": "independent option scorer",
  "pooling": "mean",
  "encoding": "pair",
  "decoder": "typed",
  "temperature": [
    0.49462966431923044,
    54.59815003313503,
    54.59815003313503
  ],
  "temperature_by_options": {},
  "loss": "supervised CE + ordinal score loss",
  "encoder_lr": 0.0001,
  "head_lr": 0.0003,
  "epochs_completed": 2,
  "seed": 42,
  "optimizer": "adamw",
  "device": "cpu",
  "precision": "fp32",
  "parameter_count": 62465
}
```

This release uses the validation-selected epoch-2 checkpoint with temperature
calibration fitted only on the separate calibration split.

## Inference

```bash
# In a checkout of advai-notebooks:
python -m pip install -e '.[decision]'
```

```python
from advai_notebooks.decision.model import DecisionModel

model = DecisionModel.load("widebluesky/wbs-decision-demo", revision="main", device="cpu")
result = model.system_one("The service is down and I will cancel.", task="triage")
print(result)
```

`triage` contains `department` (choice), `urgency` (score) and `churn_risk` (noul).
Task definitions and calibration parameters are included. Plain Transformers
`AutoModel` does not reconstruct this project's decision architecture.

Pin a commit SHA in `revision` for reproducible inference.

## Synthetic held-out test metrics

These results describe the published weights on synthetic examples only.

```json
{
  "n": 12,
  "accuracy": 0.3333333333333333,
  "nll": 0.9632161831938105,
  "brier": 0.6109643744562393,
  "ece": 0.10845711330572765,
  "normalized_score_mae": 0.3750126250088215
}
```

Only inference weights, tokenizer, model configuration and required runtime
metadata are included. Raw training records, optimizer state, checkpoints for
resuming training and local machine paths are not part of this publication.
