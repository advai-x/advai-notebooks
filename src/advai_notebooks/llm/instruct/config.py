"""Task-owned instruct configuration and supervision settings."""

from dataclasses import dataclass

from advai_notebooks.common.configuration import RuntimeConfig
from advai_notebooks.llm.configuration import (
    ExperimentSettings,
    GenerationSettings,
    ModelSettings,
    TrainingSettings,
    load_settings,
)


@dataclass
class ModelConfig(ModelSettings):
    pass


@dataclass
class DataConfig:
    samples: str = "data/sample/llm/instruct.jsonl"
    reject_truncation: bool = True


@dataclass
class TrainingConfig(TrainingSettings):
    pass


@dataclass
class EvaluationConfig:
    selection_metric: str = "nll"
    generation_metrics: list[str] | None = None


@dataclass
class ExperimentConfig(ExperimentSettings):
    pass


def load_config(path, root=None):
    cfg = load_settings(
        path,
        "llm/instruct",
        {
            "model": ModelConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "runtime": RuntimeConfig,
            "evaluation": EvaluationConfig,
            "generation": GenerationSettings,
        },
        ExperimentConfig,
        root,
    )
    metrics = cfg.evaluation.generation_metrics
    if metrics is not None and (
        not isinstance(metrics, list)
        or any(not isinstance(m, str) for m in metrics)
        or len(set(metrics)) != len(metrics)
        or any(m not in {"exact_match", "json_validity"} for m in metrics)
    ):
        raise ValueError("generation_metrics support exact_match and json_validity")
    return cfg
