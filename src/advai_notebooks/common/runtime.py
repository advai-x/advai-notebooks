"""Task-neutral single-device setup and precision handling."""

import random
from contextlib import nullcontext


def resolve_device(requested):
    import torch

    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(requested)


def setup(cfg):
    import numpy as np
    import torch

    device = resolve_device(cfg.runtime.device)
    if device.type != "cuda" and cfg.runtime.precision != "fp32":
        raise ValueError("CPU/MPS use fp32; mixed precision requires CUDA")
    if cfg.runtime.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.set_num_threads(cfg.runtime.num_threads)
    return device


def amp_context(device, precision):
    import torch

    return (
        nullcontext()
        if precision == "fp32"
        else torch.autocast(
            device_type=device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16
        )
    )
