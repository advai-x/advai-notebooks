"""Instruct next-token cross entropy over task-prepared supervised positions."""

import torch.nn.functional as F

from advai_notebooks.llm.moe import auxiliary


def statistics(logits, labels):
    shifted = labels[..., 1:].contiguous()
    count = shifted.ne(-100).sum()
    if int(count) == 0:
        raise ValueError("Batch has no supervised next-token targets")
    total = F.cross_entropy(
        logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return total, count


def loss(outputs, labels, num_items_in_batch=None, *, aux_loss_coef=None):
    total, count = statistics(outputs.logits, labels)
    # Trainer supplies the total target count across this accumulation window.
    denominator = count if num_items_in_batch is None else num_items_in_batch
    return auxiliary(outputs, total / denominator, count, denominator, aux_loss_coef)
