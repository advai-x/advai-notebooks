"""Fixed per-record noise/time draws and deterministic VAE mode for comparable evaluation."""

import torch

from .data import pixels
from .losses import per_image_mse, target
from .metrics import compute
from .sampling import evaluation_generator, noise_and_time


def evaluate(model, prepared, cfg, split="test"):
    if split not in {"validation", "test"} or not prepared[split]:
        raise ValueError("Evaluation requires a nonempty validation/test split")
    model.pipeline.unet.eval()
    predictions = []
    # Evaluate one image at a time so batch size cannot alter fixed validation draws.
    with torch.no_grad():
        for row in prepared[split]:
            latent = model.backbone.latents(
                pixels([row], model.settings.resolution, model.device), deterministic=True
            )
            text = model.backbone.text_features([row["caption"]])
            errors = []
            for draw in range(cfg.evaluation.noise_draws):
                noise, steps = noise_and_time(
                    latent,
                    model.noise_scheduler,
                    generator=evaluation_generator(cfg.evaluation.seed, row["id"], draw),
                )
                noisy = model.noise_scheduler.add_noise(latent, noise, steps)
                prediction = model.pipeline.unet(noisy, steps, encoder_hidden_states=text).sample
                errors.append(
                    float(
                        per_image_mse(
                            prediction, target(model.noise_scheduler, latent, noise, steps)
                        )[0]
                    )
                )
            predictions.append({"id": row["id"], "denoising_mse": sum(errors) / len(errors)})
    return {
        "schema_version": 1,
        "kind": "diffusion",
        "split": split,
        "examples": len(predictions),
        "metrics": compute(predictions),
        "predictions": predictions,
        "noise_draws_per_image": cfg.evaluation.noise_draws,
    }
