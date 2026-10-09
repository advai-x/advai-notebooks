"""Masked action-space MSE in train-normalized coordinates."""

import torch


def loss(model, rows):
    predicted = model.model(model.inputs(rows), model.normalized_state(rows))
    target = torch.tensor([r["actions"] for r in rows], device=model.device, dtype=torch.float32)
    mean = torch.tensor(model.statistics["action_mean"], device=model.device)
    scale = torch.tensor(model.statistics["action_std"], device=model.device)
    mask = torch.tensor([r["mask"] for r in rows], device=model.device).unsqueeze(-1)
    squared = (predicted - (target - mean) / scale).square()
    return (squared * mask).sum() / (mask.sum() * model.settings.action_dim)
