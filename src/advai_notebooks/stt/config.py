"""Strict, single-device Whisper supervised fine-tuning settings."""

from dataclasses import dataclass

from advai_notebooks.common.configuration import RuntimeConfig, positive_int, positive_number
from advai_notebooks.common.task_configuration import TrainingSettings, load_task


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    language: str | None = None
    max_new_tokens: int = 12
    max_label_tokens: int = 64


@dataclass
class DataConfig:
    samples: str = "data/sample/stt/samples.jsonl"
    min_duration: float = 0.05
    max_duration: float = 1.0


@dataclass
class TrainingConfig(TrainingSettings):
    method: str = "full"


@dataclass
class EvaluationConfig:
    selection_metric: str = "cer"
    normalization: str = "nfkc_lower_whitespace"


def validate_config(cfg):
    if not isinstance(cfg.model.id, str) or not cfg.model.id.strip():
        raise ValueError("model.id must be nonempty")
    if cfg.model.language is not None and not isinstance(cfg.model.language, str):
        raise ValueError("language must be text or null")
    if cfg.model.id != "tiny" and (not cfg.model.language or not cfg.model.language.strip()):
        raise ValueError("Set an explicit Whisper transcription language for supervised training")
    for field in ("max_new_tokens", "max_label_tokens"):
        positive_int(field, getattr(cfg.model, field))
    for field in ("min_duration", "max_duration"):
        positive_number(field, getattr(cfg.data, field))
    if not cfg.data.min_duration <= cfg.data.max_duration <= 30:
        raise ValueError("Whisper supports bounded clips up to 30 seconds, without truncation")
    if cfg.model.id == "tiny" and cfg.data.max_duration > 1:
        raise ValueError("The tiny smoke encoder supports clips up to 1 second")
    if cfg.training.method != "full":
        raise ValueError("STT currently supports full supervised fine-tuning only")
    if cfg.evaluation.selection_metric not in {"nll", "cer", "wer"}:
        raise ValueError("STT selection_metric must be nll, cer or wer")
    if cfg.evaluation.normalization not in {"none", "nfkc_lower_whitespace"}:
        raise ValueError("Unknown transcription normalization")


def load_config(path, root=None):
    return load_task(
        path,
        "stt",
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
