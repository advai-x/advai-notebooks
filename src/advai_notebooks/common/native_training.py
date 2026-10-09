"""Task-neutral epoch execution with injected losses, metrics and model persistence."""

import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from .artifacts import digest, write_json
from .checkpoint import capture_rng, restore_rng
from .provenance import source_fingerprint
from .runtime import setup
from .task_configuration import validate_native


def environment():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "peft",
        "numpy",
        "Pillow",
        "safetensors",
        "PyYAML",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "packages": packages}


def batches(rows, size, shuffle=False):
    order = torch.randperm(len(rows)).tolist() if shuffle else list(range(len(rows)))
    return [[rows[i] for i in order[start : start + size]] for start in range(0, len(rows), size)]


def train_epoch(model, rows, cfg, optimizer, loss):
    model.model.train()
    batch_list = batches(rows, cfg.training.batch_size, shuffle=True)
    total, count, steps = 0.0, 0, 0
    for start in range(0, len(batch_list), cfg.training.gradient_accumulation):
        group = batch_list[start : start + cfg.training.gradient_accumulation]
        optimizer.zero_grad(set_to_none=True)
        weights = [model.loss_weight(batch) for batch in group]
        denominator = sum(weights)
        if denominator <= 0:
            raise ValueError("Training batch has no supervised targets")
        for batch, weight in zip(group, weights):
            value = loss(model, batch)
            if value.ndim or not torch.isfinite(value):
                raise ValueError("Loss must be a finite scalar")
            (value * weight / denominator).backward()
            total += float(value.detach()) * weight
            count += weight
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.model.parameters() if p.requires_grad],
            cfg.training.max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        steps += 1
    return {"loss": total / count, "n": count, "optimizer_steps": steps}


def save_checkpoint(output, epoch, model, optimizer, state):
    parent = output / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"checkpoint-{epoch}"
    if destination.exists():
        raise ValueError(f"Checkpoint already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=parent) as tmp:
        staging = Path(tmp) / "complete"
        model.save(staging / "model")
        torch.save(
            {"optimizer": optimizer.state_dict(), "rng": capture_rng(model.device)},
            staging / "training_state.pt",
        )
        write_json(staging / "state.json", state)
        staging.rename(destination)
    write_json(
        output / "latest" / "state.json", {"checkpoint": str(destination.relative_to(output))}
    )
    write_json(output / "history.json", state["history"])


def run(cfg, model_class, load_data, prepare_data, loss, evaluate, *, resume=None, callbacks=()):
    validate_native(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Native baselines support single-process training only")
    device = setup(cfg)
    rows, summary = load_data(cfg)
    output = cfg.path(cfg.output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new path or resume")
    model = (
        model_class.load(output / "baseline", device, trainable=True)
        if resume
        else (model_class(cfg.model, device, cfg.root, cfg.training))
    )
    prepared = prepare_data(rows, model, cfg, fit=not bool(resume))
    settings = cfg.to_dict()
    settings.pop("root")
    settings.pop("output_dir")
    signature = digest(
        {
            "config": settings,
            "data": summary,
            "environment": environment(),
            "resolved_device": str(device),
            "source": source_fingerprint(cfg.kind),
        }
    )
    state = None
    if resume:
        if cfg.path(str(resume)) != output / "latest":
            raise ValueError("Resume must use the original output_dir/latest")
        pointer = json.loads((output / "latest/state.json").read_text())
        checkpoint = (output / pointer["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(output / "checkpoints"):
            raise ValueError("Invalid checkpoint pointer")
        state = json.loads((checkpoint / "state.json").read_text())
        if state["signature"] != signature:
            raise ValueError("Resume configuration/data/environment/source mismatch")
        model = model_class.load(checkpoint / "model", device, trainable=True)
        prepared = prepare_data(rows, model, cfg, fit=False)
    parameters = [p for p in model.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer_class = torch.optim.AdamW if cfg.training.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        parameters, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    if resume:
        tensors = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(tensors["optimizer"])
        restore_rng(tensors["rng"], device)
    else:
        output.mkdir(parents=True, exist_ok=True)
        model.save(output / "baseline")
        validation = evaluate(model, prepared, cfg, "validation")
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "environment.json", environment())
        write_json(output / "data_summary.json", summary)
        write_json(output / "baseline_validation.json", validation)
        write_json(output / "baseline_test.json", evaluate(model, prepared, cfg, "test"))
        state = {
            "signature": signature,
            "epoch": 0,
            "best_epoch": 0,
            "best_metric": validation["metrics"][cfg.evaluation.selection_metric],
            "history": [],
        }
        save_checkpoint(output, 0, model, optimizer, state)
    for epoch in range(state["epoch"] + 1, cfg.training.epochs + 1):
        training = train_epoch(model, prepared["train"], cfg, optimizer, loss)
        validation = evaluate(model, prepared, cfg, "validation")
        metric = validation["metrics"][cfg.evaluation.selection_metric]
        if metric < state["best_metric"]:
            state.update(best_metric=metric, best_epoch=epoch)
        state["epoch"] = epoch
        state["history"].append({"epoch": epoch, "training": training, "validation": validation})
        save_checkpoint(output, epoch, model, optimizer, state)
        for callback in callbacks:
            callback(epoch, state)
    model.save(output / "latest_export")
    selected = model_class.load(
        output / "checkpoints" / f"checkpoint-{state['best_epoch']}" / "model", device
    )
    selected_prepared = prepare_data(rows, selected, cfg, fit=False)
    selected.save(output / "export")
    report = {
        "kind": cfg.kind,
        "data": summary,
        "training": state,
        "baseline_test": json.loads((output / "baseline_test.json").read_text()),
        "finetuned_test": evaluate(selected, selected_prepared, cfg, "test"),
        "export": str(output / "export"),
        "latest_export": str(output / "latest_export"),
        "note": "Validation selects including epoch zero; test is report-only. "
        "Synthetic smoke data does not establish real-world quality.",
    }
    write_json(output / "report.json", report)
    return report
