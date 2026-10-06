"""Neutral LLM settings and validation; task data and metrics stay task-owned."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from advai_notebooks.common.configuration import (
    RuntimeConfig,
    positive_int,
    positive_number,
)
from advai_notebooks.common.configuration import (
    TrainingConfig as RuntimeTrainingConfig,
)


@dataclass
class ModelSettings:
    id: str = "tiny"
    revision: str | None = None
    max_len: int = 128
    chat_template: str | None = None


@dataclass
class TrainingSettings(RuntimeTrainingConfig):
    learning_rate: float = 5e-4
    batch_size: int = 2
    method: str = "full"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] | None = None


@dataclass
class GenerationSettings:
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class ExperimentSettings:
    kind: str
    model: ModelSettings
    data: object
    training: TrainingSettings
    runtime: RuntimeConfig
    evaluation: object
    generation: GenerationSettings
    output_dir: str
    root: Path

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def to_dict(self):
        return {**asdict(self), "root": str(self.root)}


def load_settings(path, kind, sections, cls, root=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - (set(sections) | {"output_dir"}):
        raise ValueError(f"Unknown or invalid {kind} configuration sections")
    values = {}
    for name, section_cls in sections.items():
        section = raw.get(name, {})
        if not isinstance(section, dict) or set(section) - {f.name for f in fields(section_cls)}:
            raise ValueError(f"Unknown or invalid {kind}.{name} configuration")
        values[name] = section_cls(**section)
    root = (
        Path(root).resolve()
        if root
        else next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    )
    cfg = cls(
        kind=kind, root=root, output_dir=raw.get("output_dir", f"outputs/{kind}/run"), **values
    )
    validate(cfg)
    return cfg


def validate(cfg):
    for name, value in (
        ("model.id", cfg.model.id),
        ("output_dir", cfg.output_dir),
        ("data.samples", cfg.data.samples),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if cfg.model.revision is not None and (
        not isinstance(cfg.model.revision, str) or not cfg.model.revision.strip()
    ):
        raise ValueError("model.revision must be a nonempty string")
    if cfg.model.chat_template is not None and (
        not isinstance(cfg.model.chat_template, str) or not cfg.model.chat_template.strip()
    ):
        raise ValueError("model.chat_template must be nonempty Jinja text")
    for name in ("max_len",):
        positive_int(name, getattr(cfg.model, name))
    if cfg.model.max_len < 2:
        raise ValueError("max_len must allow at least one next-token target")
    for name in ("epochs", "batch_size", "gradient_accumulation", "lora_rank", "lora_alpha"):
        positive_int(name, getattr(cfg.training, name))
    positive_int("num_threads", cfg.runtime.num_threads)
    if not isinstance(cfg.runtime.device, str) or cfg.runtime.device not in {
        "auto",
        "cpu",
        "mps",
        "cuda",
    }:
        raise ValueError("device must be auto, cpu, mps or cuda")
    if not isinstance(cfg.runtime.precision, str) or cfg.runtime.precision not in {
        "fp32",
        "fp16",
        "bf16",
    }:
        raise ValueError("precision must be fp32, fp16 or bf16")
    if not isinstance(cfg.training.optimizer, str) or cfg.training.optimizer not in {
        "adamw",
        "adam",
    }:
        raise ValueError("optimizer must be adamw or adam")
    if not isinstance(cfg.training.method, str) or cfg.training.method not in {"full", "lora"}:
        raise ValueError("method must be full or lora; QLoRA is not implemented")
    for name in ("learning_rate", "max_grad_norm"):
        positive_number(name, getattr(cfg.training, name))
    positive_number("weight_decay", cfg.training.weight_decay, True)
    positive_number("lora_dropout", cfg.training.lora_dropout, True)
    if cfg.training.lora_dropout >= 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    modules = cfg.training.lora_target_modules
    if modules is not None and (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(m, str) or not m.strip() for m in modules)
        or len(set(modules)) != len(modules)
    ):
        raise ValueError("lora_target_modules must be distinct nonempty module names")
    if type(cfg.training.seed) is not int or not 0 <= cfg.training.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for flag in (
        cfg.training.gradient_checkpointing,
        cfg.generation.do_sample,
        cfg.data.reject_truncation,
    ):
        if type(flag) is not bool:
            raise ValueError("Boolean settings must be true or false")
    positive_int("max_new_tokens", cfg.generation.max_new_tokens)
    positive_number("temperature", cfg.generation.temperature)
    positive_number("top_p", cfg.generation.top_p)
    if cfg.generation.top_p > 1:
        raise ValueError("top_p must be in (0, 1]")
    if cfg.evaluation.selection_metric != "nll":
        raise ValueError("First-version checkpoint selection uses validation nll")
