"""One-step and recursively predicted rollout errors in original state units."""

import torch

from advai_notebooks.common.native_training import batches


def evaluate(model, prepared, cfg, split="test"):
    model.model.eval()
    first_error, rollout_error, first_count, rollout_count = 0.0, 0.0, 0, 0
    with torch.inference_mode():
        for rows in batches(prepared[split], cfg.training.batch_size):
            batch = model.batch(rows)
            predicted = model.denormalize_observation(
                model.normalized_rollout(batch["observation"], batch["actions"])
            )
            squared = (predicted - batch["targets"]).square()
            mask = batch["mask"].unsqueeze(-1)
            first_error += float(squared[:, 0].sum())
            first_count += len(rows) * model.settings.observation_dim
            rollout_error += float((squared * mask).sum())
            rollout_count += int(mask.sum()) * model.settings.observation_dim
    metrics = {"state_mse": first_error / first_count, "rollout_mse": rollout_error / rollout_count}
    return {
        "split": split,
        "metrics": metrics,
        "n_state_values": first_count,
        "n_rollout_values": rollout_count,
        "horizon": cfg.model.horizon,
        "conditioning": "predicted state, ground-truth action sequence",
        "metric_records": [
            {
                "name": k,
                "value": v,
                "n": first_count if k == "state_mse" else rollout_count,
                "group": "overall",
                "direction": "minimize",
            }
            for k, v in metrics.items()
        ],
    }
