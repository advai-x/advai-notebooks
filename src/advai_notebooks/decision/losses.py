"""Supervised typed decisions and optional Gaussian-logit RLCD experiments."""


class TypedDecisionObjective:
    def __init__(self, config):
        self.config = config

    def __call__(self, logits, target, mask, qtype, sigma=1.0):
        return decision_loss(logits, target, mask, qtype, self.config, sigma)


def build_objective(config):
    if config.loss not in {"supervised", "ce", "rlcd"}:
        raise ValueError(f"Unsupported decision objective: {config.loss}")
    return TypedDecisionObjective(config)


def decision_loss(logits, target, mask, qtype, config, sigma):
    import torch

    logits = logits.float().masked_fill(~mask, -1e4)
    ce = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
    ordinal = logits.new_zeros(())
    if config.loss != "ce" and config.ordinal_weight > 0:
        # Average over all examples, so uneven accumulation groups keep the same
        # weighting. Nominal rows contribute zero to this supervised ordinal term.
        ordinal = (ordinal_error(torch.softmax(logits, -1), target, mask) * (qtype == 1)).mean()
    rl = logits.new_zeros(())
    if config.loss == "rlcd" and config.rl_weight > 0:
        # Noise and its proper-scoring advantages are detached; gradients flow
        # through the log-density of the sampled logit policy.
        with torch.no_grad():
            noise = torch.randn((config.noise_samples,) + logits.shape, device=logits.device)
            noise = noise * mask
            noise = (noise - noise.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)) * mask
            sampled = logits.unsqueeze(0) + sigma * noise
            probabilities = torch.softmax(sampled.masked_fill(~mask, -1e4), -1)
            reward = proper_reward(probabilities, target.unsqueeze(0), qtype, mask)
            advantage = reward - reward.mean(0, keepdim=True)
            advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-6)
        log_density = -(((sampled - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
        rl = -(advantage * log_density).mean()
    return config.ce_weight * ce + config.ordinal_weight * ordinal + config.rl_weight * rl


def proper_reward(probabilities, target, qtype, mask):
    """Negative Brier for nominal labels; normalized ranked probability score for ordinal labels.

    Both objectives have their maximum at the target distribution. Ordinal CDF
    distances additionally penalize mistakes farther from the correct score level.
    """
    import torch

    differences = (probabilities - target) * mask
    brier = -(differences.square()).sum(-1)
    ordinal = -ordinal_error(probabilities, target, mask)
    return torch.where(qtype == 1, ordinal, brier)


def ordinal_error(probabilities, target, mask):
    """Per-example normalized ranked probability score; final/padded CDFs excluded."""
    import torch

    differences = (probabilities - target) * mask
    counts = mask.sum(-1)
    ordinal_mask = torch.arange(mask.shape[-1], device=mask.device) < (counts - 1).unsqueeze(-1)
    return (differences.cumsum(-1).square() * ordinal_mask).sum(-1) / (counts - 1).clamp_min(1)
