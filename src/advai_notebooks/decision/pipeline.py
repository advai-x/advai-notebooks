"""Baseline -> fine-tune -> calibrate -> held-out comparison -> export."""

import importlib.metadata
import json
import platform
import subprocess
from collections import Counter

from advai_notebooks.common.artifacts import digest, write_json
from advai_notebooks.common.provenance import source_fingerprint

from .config import load_config
from .data import load_data, prepare_training_items, summary
from .evaluate import evaluate, evaluate_selected
from .model import DecisionModel
from .runtime import setup
from .schema import SPLITS
from .train import train_model


def run_experiment(config_path, root=None, resume=None):
    cfg = load_config(config_path, root)
    records, tasks = load_data(cfg.path(cfg.data.path), cfg.path(cfg.data.tasks))
    required = SPLITS if cfg.calibration.enabled else {"train", "validation", "test"}
    missing = required - {r["split"] for r in records}
    if missing:
        raise ValueError(f"Full pipeline requires configured splits; missing {sorted(missing)}")
    output = cfg.path(cfg.output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(
            f"Output directory is not empty: {output}. Choose a new output_dir or resume."
        )
    if resume is not None:
        resume = cfg.path(str(resume))
        if resume != output / "latest":
            raise ValueError("Resume from this experiment's output_dir/latest")
    device = setup(cfg)
    resume_state = None
    if resume:
        import torch

        resume_state = torch.load(
            resume / "training_state.pt", map_location="cpu", weights_only=True
        )
        model = DecisionModel.load(resume, device, weights=resume_state["model_weights"])
    else:
        model = DecisionModel.from_config(cfg, tasks, device)
    position_limit = getattr(model.model.encoder.config, "max_position_embeddings", None)
    if position_limit and cfg.model.max_len > position_limit:
        raise ValueError(f"max_len exceeds encoder position limit {position_limit}")
    model.config.update({"max_len": cfg.model.max_len, "head_max_len": cfg.model.head_max_len})
    items = prepare_training_items(records, tasks, model, cfg.data.reject_truncation)
    splits = {split: [item for item in items if item["split"] == split] for split in SPLITS}
    data_summary = summary(records, tasks)
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "safetensors",
            "huggingface-hub",
            "numpy",
            "PyYAML",
        )
    }
    implementation = source_fingerprint("decision")
    signature = digest(
        {
            "python": platform.python_version(),
            "config": cfg.to_dict(),
            "data": data_summary["fingerprint"],
            "implementation": implementation,
            "packages": versions,
            "device": str(device),
        }
    )
    if resume:
        if resume_state["signature"] != signature:
            raise ValueError(
                "Resume configuration/data/implementation differ from the saved experiment"
            )
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        baseline_metrics, baseline = evaluate(model, splits, cfg, "test")
        write_json(output / "baseline_predictions.json", baseline)
        write_json(output / "baseline_metrics.json", baseline_metrics)
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "data_summary.json", data_summary)
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True, check=False
        )
        write_json(
            output / "environment.json",
            {
                "python": platform.python_version(),
                "packages": versions,
                "device": str(device),
                "precision": cfg.runtime.precision,
                "git_commit": git.stdout.strip() if git.returncode == 0 else None,
                "source": model.source,
                "implementation_sha256": implementation,
            },
        )
    best, training = train_model(
        cfg, model, splits["train"], splits["validation"], device, signature, resume_state
    )
    evaluation = evaluate_selected(best, splits, cfg)
    calibration = evaluation["calibration"]
    best.config["fine_tuned"] = training["best_epoch"] > 0
    export = best.save(output / "export")
    write_json(export / "tasks.json", tasks)
    write_json(export / "calibration.json", calibration or {"enabled": False})
    predictions = evaluation["predictions"]
    baseline_metrics = json.loads((output / "baseline_metrics.json").read_text())
    report = {
        "dataset": data_summary,
        "model_source": best.source,
        "best_epoch": training["best_epoch"],
        "training": training,
        "baseline": baseline_metrics,
        "fine_tuned": evaluation["fine_tuned"],
        "fine_tuned_calibrated": evaluation["fine_tuned_calibrated"],
        "calibration": calibration,
        "truncated_decisions_by_split": dict(Counter(i["split"] for i in items if i["truncated"])),
        "export_dir": str(export),
        "sample_data_only": cfg.path(cfg.data.path)
        == cfg.path("data/sample/decision/triage.jsonl"),
        "random_model": cfg.model.id == "tiny",
    }
    write_json(output / "predictions.json", predictions)
    write_json(output / "report.json", report)
    return report
