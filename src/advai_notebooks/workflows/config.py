"""Combined retrieval evaluation has its own configuration."""

from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import yaml

from advai_notebooks.common import configuration as base


@dataclass
class WorkflowConfig:
    data: base.BenchmarkDataConfig
    runtime: base.RuntimeConfig
    retrieval: base.RetrievalConfig
    evaluation: base.EvaluationConfig
    batch_size: int
    seed: int
    root: Path

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def training(self):
        return SimpleNamespace(batch_size=self.batch_size, seed=self.seed)


def load_config(path, root=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    specs = {
        "data": base.BenchmarkDataConfig,
        "runtime": base.RuntimeConfig,
        "retrieval": base.RetrievalConfig,
        "evaluation": base.EvaluationConfig,
    }
    if not isinstance(raw, dict) or set(raw) - (set(specs) | {"batch_size", "seed"}):
        raise ValueError("Unknown workflow configuration")
    values = {}
    for name, cls in specs.items():
        section = raw.get(name, {})
        if not isinstance(section, dict) or set(section) - {f.name for f in fields(cls)}:
            raise ValueError(f"Invalid workflow.{name} configuration")
        values[name] = cls(**section)
    root = (
        Path(root).resolve()
        if root
        else next((p for p in path.parents if (p / "pyproject.toml").is_file()), path.parent)
    )
    cfg = WorkflowConfig(
        **values, batch_size=raw.get("batch_size", 32), seed=raw.get("seed", 42), root=root
    )
    # Reuse neutral validation without adding model/training fields to the workflow YAML.
    neutral = base.ExperimentConfig(
        "workflow",
        base.ModelConfig(),
        cfg.data,
        base.TrainingConfig(batch_size=cfg.batch_size, seed=cfg.seed),
        cfg.runtime,
        cfg.evaluation,
        "unused",
        root,
    )
    base.validate_common(neutral)
    cfg.evaluation = neutral.evaluation
    base.validate_retrieval(cfg)
    return cfg
