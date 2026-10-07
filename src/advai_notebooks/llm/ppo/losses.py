"""Token-level PPO: terminal rewards, GAE, clipped policy/value losses."""

import torch


def sequence_mean(values, mask):
    return ((values * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def advantages_and_returns(rewards, values, mask, gamma=1.0, gae_lambda=0.95, normalize=True):
    advantages = torch.zeros_like(values)
    running = torch.zeros_like(values[:, 0])
    for t in reversed(range(values.shape[1])):
        following = mask[:, t + 1] if t + 1 < values.shape[1] else torch.zeros_like(mask[:, t])
        next_value = values[:, t + 1] if t + 1 < values.shape[1] else 0.0
        delta = rewards[:, t] + gamma * next_value * following - values[:, t]
        running = (delta + gamma * gae_lambda * running * following) * mask[:, t]
        advantages[:, t] = running
    returns = (advantages + values) * mask
    if normalize:
        active = advantages[mask]
        advantages = (
            (advantages - active.mean()) / active.std(unbiased=False).clamp_min(1e-8)
        ) * mask
    return advantages.detach(), returns.detach()


def objective(
    log_probs, old_log_probs, advantages, values, old_values, returns, entropy, mask, settings
):
    ratios = (log_probs - old_log_probs).exp()
    unclipped = ratios * advantages
    clipped = ratios.clamp(1 - settings.clip_range, 1 + settings.clip_range) * advantages
    policy_loss = -sequence_mean(torch.minimum(unclipped, clipped), mask)
    clipped_values = old_values + (values - old_values).clamp(
        -settings.value_clip_range, settings.value_clip_range
    )
    value_loss = 0.5 * sequence_mean(
        torch.maximum((values - returns).square(), (clipped_values - returns).square()), mask
    )
    entropy_mean = sequence_mean(entropy, mask)
    loss = (
        policy_loss
        + settings.value_coefficient * value_loss
        - settings.entropy_coefficient * entropy_mean
    )
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy_mean.detach()),
        "clip_fraction": float(
            sequence_mean((ratios - 1).abs().gt(settings.clip_range).float(), mask)
        ),
    }
