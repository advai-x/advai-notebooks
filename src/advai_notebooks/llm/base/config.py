"""Task-owned base configuration and supervision settings."""

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
    samples: str = "data/sample/llm/base.jsonl"
    reject_truncation: bool = False


@dataclass
class TrainingConfig(TrainingSettings):
    pass


@dataclass
class EvaluationConfig:
    selection_metric: str = "nll"


@dataclass
class ExperimentConfig(ExperimentSettings):
    pass


def load_config(path, root=None):
    cfg = load_settings(
        path,
        "llm/base",
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

    return cfg
