"""Held-out action errors in the exported action coordinate system."""

import torch

from advai_notebooks.common.native_training import batches


def evaluate(model, prepared, cfg, split="test"):
    model.model.eval()
    squared, absolute, count = 0.0, 0.0, 0
    with torch.inference_mode():
        for rows in batches(prepared[split], cfg.training.batch_size):
            predicted = model.raw_actions(rows)
            target = torch.tensor(
                [r["actions"] for r in rows], device=model.device, dtype=torch.float32
            )
            mask = torch.tensor([r["mask"] for r in rows], device=model.device).unsqueeze(-1)
            squared += float(((predicted - target).square() * mask).sum())
            absolute += float(((predicted - target).abs() * mask).sum())
            count += int(mask.sum()) * model.settings.action_dim
    metrics = {"action_mse": squared / count, "action_mae": absolute / count}
    return {
        "split": split,
        "metrics": metrics,
        "n_action_values": count,
        "action_spec": model.action_spec,
        "metric_records": [
            {"name": k, "value": v, "n": count, "group": "overall", "direction": "minimize"}
            for k, v in metrics.items()
        ],
    }
