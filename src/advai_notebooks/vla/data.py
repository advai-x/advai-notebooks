"""Episode-safe action chunks and normalization fitted only on training steps."""

import math
from dataclasses import asdict

import torch

from advai_notebooks.common.artifacts import digest
from advai_notebooks.common.media import local_image, records, split_rows, text, vector


def load_data(cfg):
    path = cfg.path(cfg.data.samples)
    episodes, summary = records(path)
    rows, media, image_splits = [], [], {}
    for episode in episodes:
        text(episode.get("instruction"), "instruction")
        steps = episode.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Episode steps must be nonempty")
        last_time = None
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError("Step must be an object")
            timestamp = vector([step.get("timestamp")], 1, "timestamp")[0]
            if last_time is not None and timestamp <= last_time:
                raise ValueError("Episode timestamps must increase strictly")
            if last_time is not None and not math.isclose(
                timestamp - last_time,
                1 / cfg.data.control_frequency_hz,
                rel_tol=1e-4,
                abs_tol=1e-6,
            ):
                raise ValueError("Timestamps differ from control_frequency_hz")
            last_time = timestamp
            vector(step.get("state"), cfg.model.state_dim, "state")
            vector(step.get("action"), cfg.model.action_dim, "action")
            image, checksum = local_image(path.parent, step.get("image"))
            if image_splits.setdefault(checksum, episode["split"]) != episode["split"]:
                raise ValueError("Duplicate decoded image crosses splits")
            media.append(checksum)
            remaining = steps[i : i + cfg.model.horizon]
            actions = [vector(s.get("action"), cfg.model.action_dim, "action") for s in remaining]
            n = len(actions)
            rows.append(
                {
                    "id": f"{episode['id']}:{i}",
                    "split": episode["split"],
                    "instruction": episode["instruction"],
                    "image": str(image),
                    "state": step["state"],
                    "actions": actions + [[0.0] * cfg.model.action_dim] * (cfg.model.horizon - n),
                    "mask": [1] * n + [0] * (cfg.model.horizon - n),
                }
            )
    summary["steps"] = len(rows)
    summary["media_fingerprint"] = digest(media)
    return rows, summary


def prepare_data(rows, model, cfg, *, fit=False):
    for key in ("state_dim", "action_dim", "horizon"):
        if getattr(cfg.model, key) != getattr(model.settings, key):
            raise ValueError(f"Export/config dimensions differ: {key}")
    spec = {k: v for k, v in asdict(cfg.data).items() if k != "samples"}
    spec.update(horizon=cfg.model.horizon, action_dim=cfg.model.action_dim)
    prepared = split_rows(rows)
    if fit:
        train = prepared["train"]
        states = torch.tensor([r["state"] for r in train], dtype=torch.float32)
        # First action only: overlapping chunks must not reweight normalization.
        actions = torch.tensor([r["actions"][0] for r in train], dtype=torch.float32)
        model.statistics = {
            "state_mean": states.mean(0).tolist(),
            "state_std": states.std(0, unbiased=False).clamp_min(1e-6).tolist(),
            "action_mean": actions.mean(0).tolist(),
            "action_std": actions.std(0, unbiased=False).clamp_min(1e-6).tolist(),
        }
        model.action_spec = spec
    if model.statistics is None:
        raise ValueError("Training normalization is missing")
    if model.action_spec != spec:
        raise ValueError("Export/config action or state semantics differ")
    # Validate all instructions before training; do not silently truncate.
    for row in rows:
        model.inputs([row])
    return prepared
