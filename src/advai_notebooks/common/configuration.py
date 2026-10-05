"""Task-neutral settings and strict YAML loading."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    id: str = "tiny"
    revision: str | None = None
    max_len: int = 256
    query_prefix: str = ""
    document_prefix: str = ""


@dataclass
class RuntimeConfig:
    device: str = "auto"
    precision: str = "fp32"
    num_threads: int = 2


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    epochs: int = 2
    batch_size: int = 8
    gradient_accumulation: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    seed: int = 42


@dataclass
class BenchmarkDataConfig:
    corpus: str = "data/sample/retrieval/corpus.jsonl"
    queries: str = "data/sample/retrieval/queries.jsonl"
    qrels: str = "data/sample/retrieval/qrels.jsonl"
    allow_unjudged_negatives: bool = False
    enforce_document_groups: bool = False
    reject_truncation: bool = True


@dataclass
class EvaluationConfig:
    cutoffs: list[int] | None = None
    selection_metric: str = "mrr@10"


@dataclass
class RetrievalConfig:
    top_k: int = 100
    corpus_chunk_size: int = 1024
    query_chunk_size: int = 64


@dataclass
class ExperimentConfig:
    kind: str
    model: ModelConfig
    data: BenchmarkDataConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    output_dir: str
    root: Path

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def model_path(self, value):
        return str(self.path(value)) if self.path(value).is_dir() else value

    def to_dict(self):
        result = asdict(self)
        result["root"] = str(self.root)
        return result


def positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def positive_number(name, value, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def load_experiment(path, kind, sections, cls=ExperimentConfig, root=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - (set(sections) | {"output_dir"}):
        raise ValueError(
            f"Unknown {kind} sections; preparation and combined retrieval use separate configs"
        )
    root = (
        Path(root).resolve()
        if root
        else next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    )
    values = {}
    for name, section_cls in sections.items():
        section = raw.get(name, {})
        if not isinstance(section, dict) or set(section) - {f.name for f in fields(section_cls)}:
            raise ValueError(f"Unknown or invalid {kind}.{name} configuration")
        values[name] = section_cls(**section)
    cfg = cls(
        kind=kind, root=root, output_dir=raw.get("output_dir", f"outputs/{kind}/run"), **values
    )
    validate_common(cfg)
    return cfg


def validate_common(cfg):
    for name, value in (
        ("model.id", cfg.model.id),
        ("output_dir", cfg.output_dir),
        ("data.corpus", cfg.data.corpus),
        ("data.queries", cfg.data.queries),
        ("data.qrels", cfg.data.qrels),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if not isinstance(cfg.model.query_prefix, str) or not isinstance(
        cfg.model.document_prefix, str
    ):
        raise ValueError("Prefixes must be strings")
    if cfg.model.revision is not None and (
        not isinstance(cfg.model.revision, str) or not cfg.model.revision
    ):
        raise ValueError("revision must be a string or null")
    if cfg.runtime.device not in {"auto", "cpu", "cuda", "mps"} or cfg.runtime.precision not in {
        "fp32",
        "fp16",
        "bf16",
    }:
        raise ValueError("Unsupported device or precision")
    for name, value in (
        ("max_len", cfg.model.max_len),
        ("num_threads", cfg.runtime.num_threads),
        ("epochs", cfg.training.epochs),
        ("batch_size", cfg.training.batch_size),
        ("gradient_accumulation", cfg.training.gradient_accumulation),
    ):
        positive_int(name, value)
    if cfg.training.optimizer not in {"adamw", "adam"}:
        raise ValueError("training.optimizer must be adamw or adam")
    for name in ("learning_rate", "max_grad_norm"):
        positive_number(name, getattr(cfg.training, name))
    positive_number("weight_decay", cfg.training.weight_decay, True)
    if type(cfg.training.seed) is not int or not 0 <= cfg.training.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for flag in (
        cfg.data.allow_unjudged_negatives,
        cfg.data.enforce_document_groups,
        cfg.data.reject_truncation,
        cfg.training.gradient_checkpointing,
    ):
        if type(flag) is not bool:
            raise ValueError("Boolean settings must be true or false")
    if cfg.evaluation.cutoffs is None:
        cfg.evaluation.cutoffs = [1, 5, 10, 50, 100]
    cutoffs = cfg.evaluation.cutoffs
    if (
        not isinstance(cutoffs, list)
        or not cutoffs
        or any(type(k) is not int or k < 1 for k in cutoffs)
        or len(set(cutoffs)) != len(cutoffs)
    ):
        raise ValueError("evaluation.cutoffs must be distinct positive integers")
    if cfg.evaluation.selection_metric not in {
        f"{m}@{k}" for m in ("recall", "mrr", "ndcg") for k in cutoffs
    }:
        raise ValueError("selection_metric must be included in evaluation cutoffs")


def validate_retrieval(cfg):
    for name in ("top_k", "corpus_chunk_size", "query_chunk_size"):
        positive_int(name, getattr(cfg.retrieval, name))
    if cfg.retrieval.top_k < max(cfg.evaluation.cutoffs):
        raise ValueError("retrieval.top_k must cover evaluation cutoffs")
