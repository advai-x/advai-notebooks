"""Embedding-only model, contrastive training and retrieval evaluation settings."""

from dataclasses import dataclass

from advai_notebooks.common import configuration as base


@dataclass
class ModelConfig(base.ModelConfig):
    pooling: str = "model"


@dataclass
class DataConfig(base.BenchmarkDataConfig):
    training_samples: str = "data/sample/retrieval/embedding_samples.json"


@dataclass
class TrainingConfig(base.TrainingConfig):
    loss: str = "mnrl"
    scale: float = 20.0
    cached_mini_batch_size: int = 2


@dataclass
class ExperimentConfig(base.ExperimentConfig):
    retrieval: base.RetrievalConfig


def load_config(path, root=None):
    cfg = base.load_experiment(
        path,
        "embedding",
        {
            "model": ModelConfig,
            "data": DataConfig,
            "training": TrainingConfig,
            "runtime": base.RuntimeConfig,
            "evaluation": base.EvaluationConfig,
            "retrieval": base.RetrievalConfig,
        },
        ExperimentConfig,
        root,
    )
    if cfg.model.pooling not in {"model", "mean", "cls"}:
        raise ValueError("pooling must be model, mean or cls")
    if cfg.training.loss not in {"mnrl", "cached_mnrl"}:
        raise ValueError("Unsupported embedding loss")
    base.positive_number("scale", cfg.training.scale)
    base.positive_int("cached_mini_batch_size", cfg.training.cached_mini_batch_size)
    if not isinstance(cfg.data.training_samples, str) or not cfg.data.training_samples.strip():
        raise ValueError("data.training_samples must be a path")
    base.validate_retrieval(cfg)
    return cfg
