"""Strict configuration mechanics for independent native training tasks."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .configuration import RuntimeConfig, positive_int, positive_number


@dataclass
class TrainingSettings:
    epochs: int = 2
    batch_size: int = 2
    gradient_accumulation: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    seed: int = 42


@dataclass
class Experiment:
    kind: str
    root: Path
    output_dir: str
    model: object
    data: object
    training: object
    runtime: object
    evaluation: object

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def to_dict(self):
        value = asdict(self)
        value["root"] = str(self.root)
        return value


def load_task(path, kind, sections, validate, root=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - (set(sections) | {"output_dir"}):
        raise ValueError(f"Unknown {kind} configuration sections")
    values = {}
    for name, cls in sections.items():
        settings = raw.get(name, {})
        if not isinstance(settings, dict) or set(settings) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown or invalid {kind}.{name} settings")
        values[name] = cls(**settings)
    root = (
        Path(root).resolve()
        if root
        else next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    )
    cfg = Experiment(kind, root, raw.get("output_dir", f"outputs/{kind}/smoke"), **values)
    validate_native(cfg)
    validate(cfg)
    return cfg


def validate_native(cfg):
    for key, value in (("output_dir", cfg.output_dir), ("data.samples", cfg.data.samples)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be nonempty text")
    for key in ("epochs", "batch_size", "gradient_accumulation"):
        positive_int(key, getattr(cfg.training, key))
    positive_int("num_threads", cfg.runtime.num_threads)
    for key in ("learning_rate", "max_grad_norm"):
        positive_number(key, getattr(cfg.training, key))
    positive_number("weight_decay", cfg.training.weight_decay, allow_zero=True)
    if cfg.training.optimizer not in {"adam", "adamw"}:
        raise ValueError("optimizer must be adam or adamw")
    if type(cfg.training.seed) is not int or not 0 <= cfg.training.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if cfg.runtime.device not in {"cpu", "auto", "cuda", "mps"}:
        raise ValueError("Unsupported device")
    if cfg.runtime.precision != "fp32":
        raise ValueError("These native baselines currently support single-device fp32 only")


__all__ = ["Experiment", "RuntimeConfig", "TrainingSettings", "load_task"]
