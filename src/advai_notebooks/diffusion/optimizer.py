"""Optimize only explicitly trainable denoiser parameters."""

import torch


def parameters(model):
    return [p for p in model.pipeline.unet.parameters() if p.requires_grad]


def build(model, cfg):
    params = parameters(model)
    if not params:
        raise ValueError("No trainable UNet parameters")
    cls = torch.optim.AdamW if cfg.training.optimizer == "adamw" else torch.optim.Adam
    return cls(params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
