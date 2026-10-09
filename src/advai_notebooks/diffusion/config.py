"""Diffusion-owned settings; the first backend is Stable Diffusion latent text-to-image."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from advai_notebooks.common.configuration import (
    RuntimeConfig,
    positive_int,
    positive_number,
)
from advai_notebooks.common.configuration import (
    TrainingConfig as CommonTrainingConfig,
)


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    resolution: int = 16


@dataclass
class DataConfig:
    samples: str = "data/sample/diffusion/samples.jsonl"


@dataclass
class TrainingConfig(CommonTrainingConfig):
    method: str = "full"
    batch_size: int = 2
    learning_rate: float = 0.001
    lora_rank: int = 4
    lora_alpha: int = 4


@dataclass
class SchedulerConfig:
    prediction_type: str | None = None
    inference: str = "ddim"


@dataclass
class GenerationConfig:
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    seed: int = 123


@dataclass
class EvaluationConfig:
    selection_metric: str = "denoising_mse"
    seed: int = 1729
    noise_draws: int = 2
    sample_count: int = 2


@dataclass
class ExperimentConfig:
    kind: str
    root: Path
    output_dir: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    scheduler: SchedulerConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def to_dict(self):
        result = asdict(self)
        result["root"] = str(self.root)
        return result


def validate_config(cfg):
    for name, value in (
        ("model.id", cfg.model.id),
        ("data.samples", cfg.data.samples),
        ("output_dir", cfg.output_dir),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be nonempty text")
    if cfg.model.revision is not None and (
        not isinstance(cfg.model.revision, str) or not cfg.model.revision
    ):
        raise ValueError("revision must be nonempty text or null")
    positive_int("resolution", cfg.model.resolution)
    if cfg.model.resolution % 8:
        raise ValueError("resolution must be divisible by 8")
    if cfg.model.id == "tiny" and cfg.model.resolution != 16:
        raise ValueError("The offline tiny fixture uses resolution 16")
    if cfg.runtime.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Unsupported device")
    if cfg.runtime.precision != "fp32":
        raise ValueError("Diffusion v1 supports single-device fp32 only")
    for name in ("epochs", "batch_size", "gradient_accumulation", "lora_rank", "lora_alpha"):
        positive_int(name, getattr(cfg.training, name))
    positive_int("num_threads", cfg.runtime.num_threads)
    if cfg.training.method not in {"full", "lora"}:
        raise ValueError("method must be full or lora; only the UNet is trainable")
    if cfg.training.optimizer not in {"adam", "adamw"}:
        raise ValueError("optimizer must be adam or adamw")
    if type(cfg.training.gradient_checkpointing) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    for name in ("learning_rate", "max_grad_norm"):
        positive_number(name, getattr(cfg.training, name))
    positive_number("weight_decay", cfg.training.weight_decay, True)
    for name, seed in (
        ("training.seed", cfg.training.seed),
        ("evaluation.seed", cfg.evaluation.seed),
        ("generation.seed", cfg.generation.seed),
    ):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32)")
    if cfg.scheduler.prediction_type not in {None, "epsilon", "v_prediction"}:
        raise ValueError("Only epsilon and v_prediction objectives are supported")
    if cfg.scheduler.inference not in {"ddim", "ddpm"}:
        raise ValueError("inference scheduler must be ddim or ddpm")
    if cfg.evaluation.selection_metric != "denoising_mse":
        raise ValueError("Checkpoint selection minimizes validation denoising_mse")
    for name in ("noise_draws", "sample_count"):
        positive_int(name, getattr(cfg.evaluation, name))
    positive_int("num_inference_steps", cfg.generation.num_inference_steps)
    positive_number("guidance_scale", cfg.generation.guidance_scale, True)


def load_config(path, root=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    sections = {
        "model": ModelConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "runtime": RuntimeConfig,
        "scheduler": SchedulerConfig,
        "generation": GenerationConfig,
        "evaluation": EvaluationConfig,
    }
    if not isinstance(raw, dict) or set(raw) - (set(sections) | {"output_dir"}):
        raise ValueError("Unknown diffusion config sections")
    values = {}
    for key, cls in sections.items():
        value = raw.get(key, {})
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown diffusion.{key} configuration")
        values[key] = cls(**value)
    root = (
        Path(root).resolve()
        if root
        else next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    )
    cfg = ExperimentConfig(
        "diffusion", root, raw.get("output_dir", "outputs/diffusion/run"), **values
    )
    validate_config(cfg)
    return cfg
