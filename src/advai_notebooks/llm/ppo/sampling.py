"""ppo prompt shuffling and on-policy completion collection."""

import torch

from advai_notebooks.llm.rollout import sample


def prompt_batches(rows, batch_size):
    order = torch.randperm(len(rows)).tolist()
    for start in range(0, len(order), batch_size):
        yield [rows[i] for i in order[start : start + batch_size]]


def collect(model, rows, settings):
    expanded = rows
    rollout = sample(
        model,
        [r["rendered_prompt"] for r in expanded],
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
    )
    return expanded, rollout
