"""Diffusion noise/inference schedules, separate from optimizer learning rates."""

from diffusers import DDIMScheduler, DDPMScheduler


def scheduler_config(config):
    # SDK bookkeeping such as _use_default_values must not override saved settings.
    return {key: value for key, value in dict(config).items() if not key.startswith("_")}


def noise_scheduler(config, prediction_type=None):
    settings = scheduler_config(config)
    if prediction_type is not None:
        settings["prediction_type"] = prediction_type
    result = DDPMScheduler.from_config(settings)
    if result.config.prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError("Unsupported scheduler prediction_type")
    return result


def inference_scheduler(config, name):
    if name not in {"ddim", "ddpm"}:
        raise ValueError("Unsupported inference scheduler")
    return (DDIMScheduler if name == "ddim" else DDPMScheduler).from_config(
        scheduler_config(config)
    )
