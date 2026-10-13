"""Duration-weighted frame BCE, excluding padding outside each original clip."""

import torch
from torch.nn.functional import binary_cross_entropy


def loss(model, batch):
    total, denominator = None, 0.0
    for row in batch:
        prediction = model.probabilities(row["features"])
        labels = torch.as_tensor(row["labels"], device=model.device)
        weights = torch.as_tensor(row["weights"], device=model.device)
        value = (binary_cross_entropy(prediction, labels, reduction="none") * weights).sum()
        total = value if total is None else total + value
        denominator += float(weights.sum())
    return total / denominator
