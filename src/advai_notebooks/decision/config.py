"""Validated experiment configuration; paths are relative to the project root."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    provider: str = "hf"
    id: str = "google-bert/bert-base-multilingual-cased"
    revision: str | None = None
    subfolder: str | None = None
    max_len: int = 512
    head_max_len: int = 192
    freeze_encoder: bool = False
    encoding: str = "pair"
    head: str = "independent"
    decoder: str = "typed"
    pooling: str = "mean"
    head_hidden_size: int | None = None
    dropout: float = 0.1


@dataclass
class DataConfig:
    path: str = "data/sample/decision/triage.jsonl"
    tasks: str = "configs/decision/tasks/triage.json"
    reject_truncation: bool = True


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    epochs: int = 2
    batch_size: int = 2
    gradient_accumulation: int = 8
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    loss: str = "supervised"
    ordinal_weight: float = 1.0
    rl_weight: float = 1.0
    ce_weight: float = 1.0
    noise_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass
class RuntimeConfig:
    device: str = "auto"
    precision: str = "fp32"
    num_threads: int = 2


@dataclass
class EvaluationConfig:
    selection_metric: str = "nll"
    include_baseline: bool = True

    @property
    def maximize(self):
        from .metrics import metric_direction

        return metric_direction(self.selection_metric) == "maximize"


@dataclass
class CalibrationConfig:
    enabled: bool = True
    min_type_samples: int = 10
    min_bucket_samples: int = 2000


@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    output_dir: str
    root: Path
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def to_dict(self) -> dict:
        obj = asdict(self)
        obj["root"] = str(self.root)
        return obj


def _section(cls, raw):
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path, root: str | Path | None = None) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {
        "model",
        "data",
        "training",
        "runtime",
        "output_dir",
        "evaluation",
        "calibration",
    }:
        raise ValueError("Expected model, data, training, runtime and output_dir sections")
    if root is None:
        root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    cfg = ExperimentConfig(
        _section(ModelConfig, raw.get("model", {})),
        _section(DataConfig, raw.get("data", {})),
        _section(TrainingConfig, raw.get("training", {})),
        _section(RuntimeConfig, raw.get("runtime", {})),
        raw.get("output_dir", "outputs/decision/hf"),
        Path(root).resolve(),
        _section(EvaluationConfig, raw.get("evaluation", {})),
        _section(CalibrationConfig, raw.get("calibration", {})),
    )
    if cfg.evaluation.selection_metric not in {"nll", "brier", "ece", "accuracy"}:
        raise ValueError("evaluation.selection_metric must be nll, brier, ece or accuracy")
    if cfg.model.provider != "hf":
        raise ValueError("Decision supports provider=hf")
    for key, supported in (("encoding", "pair"), ("head", "independent"), ("decoder", "typed")):
        if getattr(cfg.model, key) != supported:
            raise ValueError(f"model.{key} currently supports only {supported}")
    if cfg.model.pooling not in {"mean", "cls"}:
        raise ValueError("model.pooling must be mean or cls")
    if cfg.model.head_hidden_size is not None and (
        type(cfg.model.head_hidden_size) is not int or cfg.model.head_hidden_size < 1
    ):
        raise ValueError("model.head_hidden_size must be a positive integer or null")
    if (
        isinstance(cfg.model.dropout, bool)
        or not isinstance(cfg.model.dropout, (int, float))
        or not math.isfinite(cfg.model.dropout)
        or not 0 <= cfg.model.dropout < 1
    ):
        raise ValueError("model.dropout must be finite and in [0, 1)")
    for key in ("revision", "subfolder"):
        value = getattr(cfg.model, key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"model.{key} must be a nonempty string or null")
    if cfg.model.id != "tiny" and not isinstance(cfg.model.id, str):
        raise ValueError("model.id must be a checkpoint ID or local directory")
    if cfg.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("runtime.device must be auto, cpu, cuda or mps")
    if cfg.runtime.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("runtime.precision must be fp32, fp16 or bf16")
    if cfg.training.optimizer not in {"adamw", "adam"}:
        raise ValueError("training.optimizer must be adamw or adam")
    if cfg.training.loss not in {"supervised", "ce", "rlcd"}:
        raise ValueError("training.loss must be supervised, ce or rlcd")
    positive_ints = {
        "min_type_samples": cfg.calibration.min_type_samples,
        "min_bucket_samples": cfg.calibration.min_bucket_samples,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "gradient_accumulation": cfg.training.gradient_accumulation,
        "noise_samples": cfg.training.noise_samples,
        "max_len": cfg.model.max_len,
        "head_max_len": cfg.model.head_max_len,
        "num_threads": cfg.runtime.num_threads,
    }
    for key, value in positive_ints.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.model.head_max_len < 32 or cfg.model.head_max_len >= cfg.model.max_len:
        raise ValueError("Require 32 <= head_max_len < max_len")
    for key in ("encoder_lr", "head_lr", "max_grad_norm", "sigma_start", "sigma_end"):
        value = getattr(cfg.training, key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"training.{key} must be finite and positive")
    for key in ("weight_decay", "rl_weight", "ce_weight", "ordinal_weight"):
        value = getattr(cfg.training, key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if cfg.training.loss == "rlcd" and cfg.training.noise_samples < 2:
        raise ValueError("RLCD needs at least two noise samples")
    if cfg.training.loss != "rlcd" and cfg.training.ce_weight <= 0:
        raise ValueError(
            "Supervised/CE training requires positive ce_weight for all question types"
        )
    if cfg.training.ce_weight + (cfg.training.rl_weight if cfg.training.loss == "rlcd" else 0) <= 0:
        raise ValueError("At least one active loss weight must be positive")
    for value in (
        cfg.evaluation.include_baseline,
        cfg.calibration.enabled,
        cfg.model.freeze_encoder,
        cfg.data.reject_truncation,
        cfg.training.gradient_checkpointing,
    ):
        if type(value) is not bool:
            raise ValueError("Boolean settings must be true or false")
    if type(cfg.training.seed) is not int or not 0 <= cfg.training.seed < 2**32:
        raise ValueError("training.seed must be an integer in [0, 2**32)")
    for key, value in (
        ("model.id", cfg.model.id),
        ("data.path", cfg.data.path),
        ("data.tasks", cfg.data.tasks),
        ("output_dir", cfg.output_dir),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string")
    return cfg
