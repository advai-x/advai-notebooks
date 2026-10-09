"""Continuous action-chunk policy configuration."""

from dataclasses import dataclass, field

from advai_notebooks.common.configuration import RuntimeConfig, positive_int, positive_number
from advai_notebooks.common.task_configuration import TrainingSettings, load_task


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    hidden_size: int = 32
    state_dim: int = 2
    action_dim: int = 2
    horizon: int = 2
    max_len: int = 16
    freeze_backbone: bool = True


@dataclass
class DataConfig:
    samples: str = "data/sample/vla/episodes.jsonl"
    state_names: list[str] = field(default_factory=lambda: ["x", "y"])
    state_units: str = "meters"
    action_names: list[str] = field(default_factory=lambda: ["dx", "dy"])
    action_units: str = "meters"
    coordinate_frame: str = "world"
    control_frequency_hz: float = 10.0


@dataclass
class TrainingConfig(TrainingSettings):
    pass


@dataclass
class EvaluationConfig:
    selection_metric: str = "action_mse"


def validate_config(cfg):
    for key in ("hidden_size", "state_dim", "action_dim", "horizon", "max_len"):
        positive_int(key, getattr(cfg.model, key))
    if not isinstance(cfg.model.id, str) or not cfg.model.id:
        raise ValueError("model.id must be nonempty")
    if type(cfg.model.freeze_backbone) is not bool:
        raise ValueError("freeze_backbone must be boolean")
    for key, size in (("state_names", cfg.model.state_dim), ("action_names", cfg.model.action_dim)):
        names = getattr(cfg.data, key)
        if (
            not isinstance(names, list)
            or len(names) != size
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"{key} must name every dimension once")
    for key in ("state_units", "action_units", "coordinate_frame"):
        if not isinstance(getattr(cfg.data, key), str) or not getattr(cfg.data, key):
            raise ValueError(f"{key} is required")
    positive_number("control_frequency_hz", cfg.data.control_frequency_hz)
    if cfg.evaluation.selection_metric != "action_mse":
        raise ValueError("VLA checkpoint selection minimizes action_mse")


def load_config(path, root=None):
    return load_task(
        path,
        "vla",
        {
            "model": ModelConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "runtime": RuntimeConfig,
            "evaluation": EvaluationConfig,
        },
        validate_config,
        root,
    )
