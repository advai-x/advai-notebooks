"""Bounded evaluation with token-weighted task metrics, independent of Trainer."""

import torch

from advai_notebooks.common.runtime import amp_context
from advai_notebooks.llm.moe import RoutingAccumulator

from .data import collator
from .losses import statistics
from .metrics import add_generation_metrics, report


def evaluate(model, prepared, cfg, split="test", raw_rows=None, generate_metrics=True):
    rows = prepared[split]
    if not rows:
        raise ValueError("Evaluation split is empty")
    total, tokens = 0.0, 0
    routing = RoutingAccumulator(model.model.config)
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
                routing.update(output, batch["attention_mask"], count)
    finally:
        model.model.train(training)
    result = report(total, tokens, len(rows))
    result["split"] = split
    if routing.enabled:
        result["moe"] = routing.report()
    if generate_metrics and cfg.evaluation.generation_metrics:
        if raw_rows is None:
            raise ValueError("Generation metrics require original conversations")
        predictions = []
        for row in raw_rows:
            if row["split"] == split:
                predictions.append(
                    {
                        "id": row["id"],
                        "prediction": model.chat(row["messages"][:-1], do_sample=False),
                        "reference": row["messages"][-1]["content"],
                    }
                )
        result = add_generation_metrics(result, predictions, cfg.evaluation.generation_metrics)
        result["predictions"] = predictions
        result["generation_protocol"] = {
            "do_sample": False,
            "max_new_tokens": model.generation.max_new_tokens,
        }
    return result
