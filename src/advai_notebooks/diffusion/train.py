"""Denoiser full/LoRA training with example-weighted gradient accumulation."""

import torch

from .data import pixels
from .losses import per_image_mse, target
from .optimizer import parameters
from .sampling import batches, noise_and_time


def train_epoch(model, rows, cfg, optimizer):
    model.pipeline.unet.train()
    minibatches = batches(rows, cfg.training.batch_size, shuffle=True)
    total, examples, updates = 0.0, 0, 0
    for start in range(0, len(minibatches), cfg.training.gradient_accumulation):
        window = minibatches[start : start + cfg.training.gradient_accumulation]
        count = sum(len(batch) for batch in window)
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            latent = model.backbone.latents(pixels(batch, model.settings.resolution, model.device))
            text = model.backbone.text_features([r["caption"] for r in batch])
            noise, steps = noise_and_time(latent, model.noise_scheduler)
            noisy = model.noise_scheduler.add_noise(latent, noise, steps)
            prediction = model.pipeline.unet(noisy, steps, encoder_hidden_states=text).sample
            losses = per_image_mse(prediction, target(model.noise_scheduler, latent, noise, steps))
            if not torch.isfinite(losses).all():
                raise ValueError("Nonfinite denoising loss; checkpoint not committed")
            (losses.sum() / count).backward()
            total += float(losses.detach().sum())
            examples += len(batch)
        torch.nn.utils.clip_grad_norm_(
            parameters(model), cfg.training.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        updates += 1
    return {"denoising_mse": total / examples, "examples": examples, "optimizer_steps": updates}
