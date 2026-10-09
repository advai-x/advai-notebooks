"""Per-image denoising MSE with epsilon or velocity targets."""

import torch.nn.functional as F


def target(scheduler, latents, noise, timesteps):
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return noise
    if kind == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError("Unsupported diffusion prediction_type")


def per_image_mse(prediction, expected):
    return F.mse_loss(prediction.float(), expected.float(), reduction="none").flatten(1).mean(1)
