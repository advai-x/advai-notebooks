"""Single-speaker Qwen3-TTS Base and synthetic lifecycle-test settings."""

from dataclasses import dataclass

from advai_notebooks.common.configuration import RuntimeConfig, positive_int, positive_number
from advai_notebooks.common.task_configuration import TrainingSettings, load_task


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    max_text_tokens: int = 128
    max_new_tokens: int = 16


@dataclass
class DataConfig:
    samples: str = "data/sample/tts/samples.jsonl"
    reference_audio: str = "audio/one.wav"
    min_duration: float = 0.05
    max_duration: float = 10.0


@dataclass
class TrainingConfig(TrainingSettings):
    method: str = "full"


@dataclass
class EvaluationConfig:
    selection_metric: str = "loss"


def validate_config(cfg):
    if not isinstance(cfg.model.id, str) or not cfg.model.id.strip():
        raise ValueError("model.id must be nonempty")
    if not isinstance(cfg.data.reference_audio, str) or not cfg.data.reference_audio:
        raise ValueError("reference_audio must identify a training-only reference clip")
    for field in ("max_text_tokens", "max_new_tokens"):
        positive_int(field, getattr(cfg.model, field))
    for field in ("min_duration", "max_duration"):
        positive_number(field, getattr(cfg.data, field))
    if cfg.data.min_duration > cfg.data.max_duration:
        raise ValueError("Invalid audio duration bounds")
    if cfg.training.method != "full":
        raise ValueError("TTS currently supports full talker fine-tuning only")
    if cfg.evaluation.selection_metric != "loss":
        raise ValueError("TTS selection_metric must be loss; quality requires listening evaluation")


def load_config(path, root=None):
    return load_task(
        path,
        "tts",
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
