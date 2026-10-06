"""Bounded evaluation with token-weighted task metrics, independent of Trainer."""

import torch

from advai_notebooks.common.runtime import amp_context

from .data import collator
from .losses import statistics
from .metrics import report


def evaluate(model, prepared, cfg, split="test", raw_rows=None, generate_metrics=True):
    rows = prepared[split]
    if not rows:
        raise ValueError("Evaluation split is empty")
    total, tokens = 0.0, 0
    training = model.model.training
    model.model.eval()
    pad = collator(model.tokenizer)
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), cfg.training.batch_size):
                batch = {
                    k: v.to(model.device)
                    for k, v in pad(rows[start : start + cfg.training.batch_size]).items()
                }
                labels = batch.pop("labels")
                with amp_context(model.device, cfg.runtime.precision):
                    output = model.model(**batch)
                    loss, count = statistics(output.logits, labels)
                total += float(loss)
                tokens += int(count)
    finally:
        model.model.train(training)
    result = report(total, tokens, len(rows))
    result["split"] = split

    return result
