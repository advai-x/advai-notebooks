"""Offline sample preparation and optional model-based candidate generation."""

from dataclasses import dataclass

from advai_notebooks.common import configuration as base


@dataclass
class ModelConfig(base.ModelConfig):
    pooling: str = "model"


@dataclass
class MiningConfig:
    strategy: str = "judged"
    negatives_per_query: int = 2
    candidate_strategy: str = "all_corpus"


@dataclass
class PreparationConfig(base.ExperimentConfig):
    retrieval: base.RetrievalConfig
    mining: MiningConfig


def load_config(path, root=None):
    cfg = base.load_experiment(
        path,
        "data_preparation",
        {
            "model": ModelConfig,
            "data": base.BenchmarkDataConfig,
            "training": base.TrainingConfig,
            "runtime": base.RuntimeConfig,
            "evaluation": base.EvaluationConfig,
            "retrieval": base.RetrievalConfig,
            "mining": MiningConfig,
        },
        PreparationConfig,
        root,
    )
    if cfg.mining.strategy not in {"judged", "embedding"} or cfg.mining.candidate_strategy not in {
        "all_corpus",
        "embedding",
    }:
        raise ValueError(
            "Mining strategy must be judged/embedding; candidate strategy all_corpus/embedding"
        )
    if type(cfg.mining.negatives_per_query) is not int or cfg.mining.negatives_per_query < 1:
        raise ValueError("Preparation requires a positive negative count")
    if cfg.model.pooling not in {"model", "mean", "cls"}:
        raise ValueError("Unsupported preparation encoder pooling")
    base.validate_retrieval(cfg)
    return cfg
