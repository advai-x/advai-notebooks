"""Masked multi-step normalized state error with recursive predicted context."""


def loss(model, rows):
    batch = model.batch(rows)
    predicted = model.normalized_rollout(batch["observation"], batch["actions"])
    targets = model.normalize(batch["targets"], "observation")
    mask = batch["mask"].unsqueeze(-1)
    return ((predicted - targets).square() * mask).sum() / (
        mask.sum() * model.settings.observation_dim
    )
