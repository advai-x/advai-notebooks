"""ppo-owned algorithm, reward and experiment configuration."""

from dataclasses import dataclass

from advai_notebooks.common.configuration import RuntimeConfig, positive_int, positive_number
from advai_notebooks.llm.configuration import (
    ExperimentSettings,
    GenerationSettings,
    ModelSettings,
    TrainingSettings,
    load_settings,
    validate,
)


@dataclass
class DataConfig:
    samples: str = "data/sample/llm/ppo.jsonl"
    reject_truncation: bool = True


@dataclass
class EvaluationConfig:
    selection_metric: str = "reward_mean"


@dataclass
class RewardConfig:
    exact_match_weight: float = 1.0
    token_overlap_weight: float = 0.0
    json_validity_weight: float = 0.0
    length_weight: float = 0.0
    target_length: int = 8
    custom_function: str | None = None


@dataclass
class AlgorithmConfig:
    update_epochs: int = 2
    minibatch_size: int = 2
    clip_range: float = 0.2
    kl_coefficient: float = 0.05
    temperature: float = 1.0
    max_new_tokens: int = 16

    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    normalize_advantages: bool = True


@dataclass
class ExperimentConfig(ExperimentSettings):
    algorithm: AlgorithmConfig
    reward: RewardConfig


def validate_config(cfg):
    validate(cfg, selection_metric="reward_mean")
    if cfg.runtime.precision != "fp32":
        raise ValueError("ppo v1 supports fp32 only; mixed precision is not implemented")
    if cfg.training.gradient_checkpointing:
        raise ValueError("ppo v1 does not support gradient checkpointing")
    if not cfg.data.reject_truncation:
        raise ValueError("RL rejects prompt truncation; reject_truncation must be true")
    if cfg.training.lora_dropout != 0:
        raise ValueError("RL uses dropout-free policy ratios; lora_dropout must be zero")
    for name in ("update_epochs", "minibatch_size", "max_new_tokens"):
        positive_int(name, getattr(cfg.algorithm, name))
    for name in ("clip_range", "temperature"):
        positive_number(name, getattr(cfg.algorithm, name))
    if cfg.algorithm.clip_range >= 1:
        raise ValueError("clip_range must be in (0, 1)")
    positive_number("kl_coefficient", cfg.algorithm.kl_coefficient, True)

    for name in ("gamma", "gae_lambda"):
        value = getattr(cfg.algorithm, name)
        positive_number(name, value, True)
        if value > 1:
            raise ValueError(f"{name} must be in [0, 1]")
    positive_number("value_clip_range", cfg.algorithm.value_clip_range)
    positive_number("value_coefficient", cfg.algorithm.value_coefficient, True)
    positive_number("entropy_coefficient", cfg.algorithm.entropy_coefficient, True)
    if type(cfg.algorithm.normalize_advantages) is not bool:
        raise ValueError("normalize_advantages must be boolean")

    for name in (
        "exact_match_weight",
        "token_overlap_weight",
        "json_validity_weight",
        "length_weight",
    ):
        positive_number(name, getattr(cfg.reward, name), True)
    positive_int("target_length", cfg.reward.target_length)
    custom = cfg.reward.custom_function
    if custom is not None and (
        not isinstance(custom, str) or custom.count(":") != 1 or not all(custom.split(":"))
    ):
        raise ValueError("custom_function must be module:function")
    if not custom and not any(
        getattr(cfg.reward, name)
        for name in (
            "exact_match_weight",
            "token_overlap_weight",
            "json_validity_weight",
            "length_weight",
        )
    ):
        raise ValueError("At least one reward must be enabled")


def load_config(path, root=None):
    return load_settings(
        path,
        "llm/ppo",
        {
            "model": ModelSettings,
            "data": DataConfig,
            "training": TrainingSettings,
            "runtime": RuntimeConfig,
            "evaluation": EvaluationConfig,
            "generation": GenerationSettings,
            "algorithm": AlgorithmConfig,
            "reward": RewardConfig,
        },
        ExperimentConfig,
        root,
        validator=validate_config,
    )
