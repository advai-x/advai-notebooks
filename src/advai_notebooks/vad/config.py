"""Bounded, single-device VAD settings; no transcription or diarization labels."""

from dataclasses import dataclass, field

from advai_notebooks.common.configuration import RuntimeConfig, positive_number
from advai_notebooks.common.task_configuration import TrainingSettings, load_task


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    sha256: str | None = None
    sample_rate: int = 16000
    threshold: float = 0.5
    exit_threshold: float = 0.35
    min_speech_duration: float = 0.064
    min_silence_duration: float = 0.096
    speech_pad: float = 0.032
    max_inference_duration: float = 300.0


@dataclass
class DataConfig:
    samples: str = "data/sample/vad/samples.jsonl"
    min_duration: float = 0.032
    max_duration: float = 30.0


@dataclass
class TrainingConfig(TrainingSettings):
    method: str = "decoder"


@dataclass
class EvaluationConfig:
    selection_metric: str = "bce"
    calibrate: bool = True
    thresholds: list = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7])
    hysteresis: float = 0.15


def probability(value, name):
    positive_number(name, value, allow_zero=True)
    if value > 1:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_model(model):
    if model.id not in {"tiny", "silero"}:
        raise ValueError("VAD supports tiny or the pinned Silero package backend")
    if type(model.sample_rate) is not int or model.sample_rate not in {8000, 16000}:
        raise ValueError("VAD sample_rate must be 8000 or 16000")
    if model.id == "silero" and (
        not isinstance(model.revision, str)
        or not model.revision.strip()
        or not isinstance(model.sha256, str)
        or len(model.sha256) != 64
        or any(c not in "0123456789abcdef" for c in model.sha256)
    ):
        raise ValueError("Pin the Silero package revision and bundled JIT sha256")
    probability(model.threshold, "threshold")
    probability(model.exit_threshold, "exit_threshold")
    if not model.exit_threshold < model.threshold:
        raise ValueError("exit_threshold must be lower than threshold")
    for name in ("min_speech_duration", "min_silence_duration", "speech_pad"):
        positive_number(name, getattr(model, name), allow_zero=True)
    positive_number("max_inference_duration", model.max_inference_duration)
    if model.max_inference_duration > 3600:
        raise ValueError("Offline detection is bounded to at most one hour")


def validate_config(cfg):
    validate_model(cfg.model)
    for name in ("min_duration", "max_duration"):
        positive_number(name, getattr(cfg.data, name))
    if not cfg.data.min_duration <= cfg.data.max_duration <= 300:
        raise ValueError("Training clips must be bounded to at most 300 seconds")
    if cfg.training.method != "decoder":
        raise ValueError("VAD trains the decoder only; the encoder remains frozen")
    if cfg.evaluation.selection_metric != "bce":
        raise ValueError("VAD selects checkpoints by validation bce")
    if type(cfg.evaluation.calibrate) is not bool:
        raise ValueError("calibrate must be boolean")
    if not isinstance(cfg.evaluation.thresholds, list) or not cfg.evaluation.thresholds:
        raise ValueError("Provide a nonempty list of threshold candidates")
    probability(cfg.evaluation.hysteresis, "hysteresis")
    if cfg.evaluation.hysteresis <= 0:
        raise ValueError("hysteresis must be positive")
    for value in cfg.evaluation.thresholds:
        probability(value, "threshold candidate")
        if value <= 0:
            raise ValueError("Threshold candidates must be positive")


def load_config(path, root=None):
    return load_task(
        path,
        "vad",
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
