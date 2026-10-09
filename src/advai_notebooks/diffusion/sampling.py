"""Task-local batch order and training/evaluation noise draws."""

import hashlib

import torch


def batches(rows, size, *, shuffle=False):
    order = torch.randperm(len(rows)).tolist() if shuffle else range(len(rows))
    ordered = [rows[i] for i in order]
    return [ordered[i : i + size] for i in range(0, len(rows), size)]


def noise_and_time(latents, scheduler, *, generator=None):
    noise = torch.randn(latents.shape, generator=generator, device="cpu").to(latents.device)
    steps = torch.randint(
        0, scheduler.config.num_train_timesteps, (len(latents),), generator=generator, device="cpu"
    ).to(latents.device)
    return noise, steps


def evaluation_generator(seed, row_id, draw):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{row_id}:{draw}".encode()).digest()[:8], "big")
    return torch.Generator(device="cpu").manual_seed(value)
