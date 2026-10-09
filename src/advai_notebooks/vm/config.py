"""Action-conditioned state-space world model configuration."""

from dataclasses import dataclass, field

from advai_notebooks.common.configuration import RuntimeConfig, positive_int, positive_number
from advai_notebooks.common.task_configuration import TrainingSettings, load_task


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    observation_dim: int = 2
    action_dim: int = 2
    hidden_size: int = 32
    horizon: int = 3
    predict_delta: bool = True


@dataclass
class DataConfig:
    samples: str = "data/sample/vm/episodes.jsonl"
    observation_names: list[str] = field(default_factory=lambda: ["x", "y"])
    action_names: list[str] = field(default_factory=lambda: ["dx", "dy"])
    observation_units: str = "meters"
    action_units: str = "meters"
    control_frequency_hz: float = 10.0


@dataclass
class TrainingConfig(TrainingSettings):
    pass


@dataclass
class EvaluationConfig:
    selection_metric: str = "rollout_mse"


def validate_config(cfg):
    positive_number("control_frequency_hz", cfg.data.control_frequency_hz)
    for key in ("observation_dim", "action_dim", "hidden_size", "horizon"):
        positive_int(key, getattr(cfg.model, key))
    if not isinstance(cfg.model.id, str) or not cfg.model.id:
        raise ValueError("model.id must be nonempty")
    if type(cfg.model.predict_delta) is not bool:
        raise ValueError("predict_delta must be boolean")
    for key, size in (
        ("observation_names", cfg.model.observation_dim),
        ("action_names", cfg.model.action_dim),
    ):
        values = getattr(cfg.data, key)
        if (
            not isinstance(values, list)
            or len(values) != size
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{key} must name every dimension once")
    for key in ("observation_units", "action_units"):
        if not isinstance(getattr(cfg.data, key), str) or not getattr(cfg.data, key):
            raise ValueError(f"{key} is required")
    if cfg.evaluation.selection_metric not in {"state_mse", "rollout_mse"}:
        raise ValueError("VM selection_metric must be state_mse or rollout_mse")


def load_config(path, root=None):
    return load_task(
        path,
        "vm",
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
