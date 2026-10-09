"""Residual world dynamics with recursive, action-conditioned state rollouts."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.hub import resolve_export
from advai_notebooks.common.media import vector

from .config import ModelConfig


class Dynamics(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(settings.observation_dim + settings.action_dim, settings.hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(settings.hidden_size, settings.observation_dim),
        )
        self.predict_delta = settings.predict_delta

    def forward(self, observation, action):
        predicted = self.network(torch.cat([observation, action], dim=-1))
        return observation + predicted if self.predict_delta else predicted


class VMModel:
    def __init__(self, settings=None, device="cpu", root=None, training=None):
        del training
        self.settings = settings or ModelConfig()
        self.device = torch.device(device)
        if self.settings.id != "tiny":
            local = Path(root or ".") / self.settings.id
            path = str(local.resolve()) if local.is_dir() else self.settings.id
            if not local.is_dir() and not self.settings.revision:
                raise ValueError("Pin model.revision when loading a Hub base")
            previous = self.load(path, device, revision=self.settings.revision)
            for key in ("observation_dim", "action_dim", "hidden_size", "predict_delta"):
                if getattr(self.settings, key) != getattr(previous.settings, key):
                    raise ValueError(f"VM base architecture differs: {key}")
            self.model = previous.model
            self.statistics, self.state_spec = previous.statistics, previous.state_spec
        else:
            self.model = Dynamics(self.settings).to(device=self.device, dtype=torch.float32)
            self.statistics, self.state_spec = None, None

    def normalize(self, value, kind):
        mean = torch.tensor(self.statistics[f"{kind}_mean"], device=self.device)
        scale = torch.tensor(self.statistics[f"{kind}_std"], device=self.device)
        return (value - mean) / scale

    def denormalize_observation(self, value):
        mean = torch.tensor(self.statistics["observation_mean"], device=self.device)
        scale = torch.tensor(self.statistics["observation_std"], device=self.device)
        return value * scale + mean

    def normalized_rollout(self, observations, actions):
        state = self.normalize(observations, "observation")
        actions = self.normalize(actions, "action")
        predictions = []
        for step in range(actions.shape[1]):
            state = self.model(state, actions[:, step])
            predictions.append(state)
        return torch.stack(predictions, dim=1)

    def batch(self, rows):
        return {
            key: torch.tensor([r[key] for r in rows], device=self.device, dtype=torch.float32)
            for key in ("observation", "actions", "targets", "mask")
        }

    def loss_weight(self, rows):
        return sum(sum(r["mask"]) * self.settings.observation_dim for r in rows)

    def rollout(self, observation, actions):
        if self.statistics is None:
            raise ValueError("Fit training normalization or load a trained export first")
        vector(observation, self.settings.observation_dim, "observation")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Rollout requires a nonempty action sequence")
        for action in actions:
            vector(action, self.settings.action_dim, "action")
        if len(actions) > self.settings.horizon:
            raise ValueError("Rollout exceeds the exported prediction horizon")
        self.model.eval()
        with torch.inference_mode():
            predicted = self.normalized_rollout(
                torch.tensor([observation], device=self.device, dtype=torch.float32),
                torch.tensor([actions], device=self.device, dtype=torch.float32),
            )
            states = self.denormalize_observation(predicted)[0].cpu().tolist()
        return {"observations": states, "state_spec": self.state_spec}

    def predict(self, observation, action):
        return self.rollout(observation, [action])["observations"][0]

    def save(self, path):
        if self.statistics is None or self.state_spec is None:
            raise ValueError("An export must include fitted normalization and state semantics")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()},
            path / "model.safetensors",
        )
        write_json(
            path / "advai_metadata.json",
            {
                "schema_version": 1,
                "kind": "vm",
                "model": asdict(self.settings),
                "statistics": self.statistics,
                "state_spec": self.state_spec,
            },
        )
        return path

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **options):
        del trainable
        path = resolve_export(path, **options)
        meta = json.loads((path / "advai_metadata.json").read_text())
        if meta.get("kind") != "vm" or meta.get("schema_version") != 1:
            raise ValueError("Not a supported World Model export")
        obj = cls.__new__(cls)
        obj.settings, obj.device = ModelConfig(**meta["model"]), torch.device(device)
        obj.model = Dynamics(obj.settings)
        obj.model.load_state_dict(load_file(path / "model.safetensors"))
        obj.model.to(device=obj.device, dtype=torch.float32)
        obj.statistics, obj.state_spec = meta["statistics"], meta["state_spec"]
        return obj
