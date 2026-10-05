"""Ranker-only settings; candidates and pairs are prepared before training."""

from dataclasses import dataclass

from advai_notebooks.common import configuration as base

ModelConfig = base.ModelConfig


@dataclass
class DataConfig(base.BenchmarkDataConfig):
    training_samples: str = "data/sample/retrieval/ranker_samples.json"
    candidates: str = "data/sample/retrieval/candidates.json"


@dataclass
class TrainingConfig(base.TrainingConfig):
    loss: str = "bce"


def load_config(path, root=None):
    cfg = base.load_experiment(
        path,
        "ranker",
        {
            "model": ModelConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "runtime": base.RuntimeConfig,
            "evaluation": base.EvaluationConfig,
        },
        root=root,
    )
    if cfg.training.loss != "bce":
        raise ValueError("Unsupported ranker loss")
    for value in (cfg.data.training_samples, cfg.data.candidates):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("data.training_samples and data.candidates must be paths")
    return cfg
