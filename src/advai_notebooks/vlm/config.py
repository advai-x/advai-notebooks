"""LLaVA image/answer training configuration."""

from dataclasses import dataclass

from advai_notebooks.common.configuration import RuntimeConfig, positive_int
from advai_notebooks.common.task_configuration import TrainingSettings, load_task


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    max_len: int = 128
    max_new_tokens: int = 8
    freeze_vision: bool = True


@dataclass
class DataConfig:
    samples: str = "data/sample/vlm/samples.jsonl"


@dataclass
class TrainingConfig(TrainingSettings):
    method: str = "full"
    lora_rank: int = 4
    lora_alpha: int = 8


@dataclass
class EvaluationConfig:
    selection_metric: str = "nll"


def validate_config(cfg):
    if not isinstance(cfg.model.id, str) or not cfg.model.id:
        raise ValueError("model.id must be nonempty")
    for field in ("max_len", "max_new_tokens"):
        positive_int(field, getattr(cfg.model, field))
    if cfg.model.max_new_tokens >= cfg.model.max_len:
        raise ValueError("max_new_tokens must leave room for the image prompt")
    if type(cfg.model.freeze_vision) is not bool:
        raise ValueError("freeze_vision must be boolean")
    if cfg.training.method not in {"full", "lora"}:
        raise ValueError("VLM supports full or lora")
    if cfg.training.method == "lora" and not cfg.model.freeze_vision:
        raise ValueError("Language-only LoRA requires freeze_vision=true")
    for field in ("lora_rank", "lora_alpha"):
        positive_int(field, getattr(cfg.training, field))
    if cfg.evaluation.selection_metric != "nll":
        raise ValueError("VLM selection_metric must be nll")


def load_config(path, root=None):
    return load_task(
        path,
        "vlm",
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
