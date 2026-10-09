"""Token-weighted held-out language loss; image prompts remain unlabelled."""

import math

import torch

from advai_notebooks.common.native_training import batches

from .losses import loss


def evaluate(model, prepared, cfg, split="test"):
    model.model.eval()
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for rows in batches(prepared[split], cfg.training.batch_size):
            n = model.loss_weight(rows)
            total += float(loss(model, rows)) * n
            tokens += n
    nll = total / tokens
    if not math.isfinite(nll):
        raise ValueError("Nonfinite evaluation loss")
    return {
        "split": split,
        "metrics": {"nll": nll, "perplexity": math.exp(min(nll, 50))},
        "n_tokens": tokens,
        "n_samples": len(prepared[split]),
        "metric_records": [
            {"name": "nll", "value": nll, "n": tokens, "group": "overall", "direction": "minimize"}
        ],
    }
