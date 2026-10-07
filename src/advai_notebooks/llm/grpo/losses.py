"""Outcome-supervised GRPO: per-prompt relative advantages, clipped ratios and KL."""

import torch


def sequence_mean(values, mask):
    # Each completion has equal weight, independently of its generated length.
    return ((values * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def group_advantages(rewards, group_size, epsilon=1e-6):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("Rewards must form complete groups of at least two completions")
    groups = rewards.reshape(-1, group_size)
    centered = groups - groups.mean(-1, keepdim=True)
    deviations = groups.std(-1, keepdim=True, unbiased=False)
    # Zero-variance groups contribute zero policy advantage, while KL remains active.
    return (centered / deviations.clamp_min(epsilon)).reshape(-1).detach()


def objective(log_probs, old_log_probs, reference_log_probs, advantages, mask, settings):
    ratios = (log_probs - old_log_probs).exp()
    advantages = advantages[:, None]
    clipped = ratios.clamp(1 - settings.clip_range, 1 + settings.clip_range) * advantages
    policy_loss = -sequence_mean(torch.minimum(ratios * advantages, clipped), mask)
    difference = reference_log_probs - log_probs
    kl = difference.expm1() - difference
    kl_mean = sequence_mean(kl, mask)
    loss = policy_loss + settings.kl_coefficient * kl_mean
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "reference_kl": float(kl_mean.detach()),
        "clip_fraction": float(
            sequence_mean((ratios - 1).abs().gt(settings.clip_range).float(), mask)
        ),
    }
