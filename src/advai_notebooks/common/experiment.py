"""Record enough run provenance to audit an experiment."""

import importlib.metadata
import platform
from pathlib import Path

from .artifacts import digest, write_json
from .provenance import source_fingerprint


def record_environment(output, cfg, adapter):
    base = adapter.backbone
    write_json(
        Path(output) / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "sentence-transformers", "numpy", "PyYAML")
            },
            "device": str(adapter.device),
            "precision": cfg.runtime.precision,
            "seed": cfg.training.seed,
            "source_sha256": source_fingerprint(cfg.kind),
            "model_id": cfg.model.id,
            "requested_revision": cfg.model.revision,
            "resolved_revision": getattr(base.config, "_commit_hash", None),
            "tiny_model": cfg.model.id == "tiny",
        },
    )


def run_signature(cfg, inputs, device):
    return digest(
        {
            "python": platform.python_version(),
            "config": cfg.to_dict(),
            "inputs": inputs,
            "source_sha256": source_fingerprint(cfg.kind),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "sentence-transformers", "numpy", "PyYAML")
            },
            "device": str(device),
        }
    )
