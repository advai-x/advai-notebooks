"""Episode-contained temporal windows and training-only state/action statistics."""

import math
from dataclasses import asdict

import torch

from advai_notebooks.common.artifacts import digest
from advai_notebooks.common.media import records, split_rows, vector


def load_data(cfg):
    episodes, summary = records(cfg.path(cfg.data.samples))
    rows, seen, train_observations, train_actions = [], {}, [], []
    for episode in episodes:
        observations, actions = episode.get("observations"), episode.get("actions")
        times = episode.get("timestamps")
        if (
            not isinstance(observations, list)
            or len(observations) < 2
            or not isinstance(actions, list)
            or len(actions) != len(observations) - 1
            or not isinstance(times, list)
            or len(times) != len(observations)
        ):
            raise ValueError("Episode requires N observations/timestamps and N-1 actions")
        vector(times, len(times), "timestamps")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Episode timestamps must increase strictly")
        if any(
            not math.isclose(b - a, 1 / cfg.data.control_frequency_hz, rel_tol=1e-4, abs_tol=1e-6)
            for a, b in zip(times, times[1:])
        ):
            raise ValueError("Timestamps differ from control_frequency_hz")
        for value in observations:
            vector(value, cfg.model.observation_dim, "observation")
        for value in actions:
            vector(value, cfg.model.action_dim, "action")
        checksum = digest({"observations": observations, "actions": actions})
        if seen.setdefault(checksum, episode["split"]) != episode["split"]:
            raise ValueError("Duplicate trajectory crosses splits")
        if episode["split"] == "train":
            train_observations.extend(observations)
            train_actions.extend(actions)
        for start in range(len(actions)):
            chunk = actions[start : start + cfg.model.horizon]
            n = len(chunk)
            targets = observations[start + 1 : start + 1 + n]
            rows.append(
                {
                    "id": f"{episode['id']}:{start}",
                    "split": episode["split"],
                    "observation": observations[start],
                    "actions": chunk + [[0.0] * cfg.model.action_dim] * (cfg.model.horizon - n),
                    "targets": targets
                    + [[0.0] * cfg.model.observation_dim] * (cfg.model.horizon - n),
                    "mask": [1] * n + [0] * (cfg.model.horizon - n),
                }
            )
    summary["windows"] = len(rows)
    # Normalization sources stay in-memory, not in exported/public summary.
    for row in rows:
        if row["split"] == "train":
            row["_fit_observations"] = train_observations
            row["_fit_actions"] = train_actions
            break
    return rows, summary


def prepare_data(rows, model, cfg, *, fit=False):
    for key in ("observation_dim", "action_dim", "horizon"):
        if getattr(cfg.model, key) != getattr(model.settings, key):
            raise ValueError(f"Export/config dimensions differ: {key}")
    spec = {k: v for k, v in asdict(cfg.data).items() if k != "samples"}
    if fit and model.statistics is None:
        sources = next(r for r in rows if r["split"] == "train")
        statistics = {}
        for kind, values in (
            ("observation", sources["_fit_observations"]),
            ("action", sources["_fit_actions"]),
        ):
            values = torch.tensor(values, dtype=torch.float32)
            statistics[f"{kind}_mean"] = values.mean(0).tolist()
            statistics[f"{kind}_std"] = values.std(0, unbiased=False).clamp_min(1e-6).tolist()
        model.statistics = statistics
        model.state_spec = spec
    if model.statistics is None:
        raise ValueError("Training normalization is missing")
    if model.state_spec != spec:
        raise ValueError("Export/config state or action semantics differ")
    return split_rows(rows)
