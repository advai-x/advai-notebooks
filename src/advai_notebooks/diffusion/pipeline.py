"""Independent baseline/train/validation selection/test/export lifecycle."""

import gc
import json

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.checkpoint import restore_rng
from advai_notebooks.common.runtime import setup

from .config import validate_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .metrics import is_improvement, select_metric
from .model import DiffusionModel
from .optimizer import build
from .runtime import (
    environment,
    resume_checkpoint,
    save_checkpoint,
    save_export,
    signature,
    single_process,
)
from .train import train_epoch


def run(cfg, resume=None, *, callbacks=()):
    validate_config(cfg)
    single_process()
    device = setup(cfg)
    rows, summary = load_data(cfg)
    output = cfg.path(cfg.output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a new path or resume")
    initial = (
        DiffusionModel.load(output / "baseline", device, trainable=True)
        if resume
        else (
            DiffusionModel(cfg.model, device, cfg.root, cfg.training, cfg.generation, cfg.scheduler)
        )
    )
    initial.configure(cfg.model, cfg.generation, cfg.scheduler)
    prepared = prepare_data(rows, initial, cfg)
    current_signature = signature(cfg, summary, initial)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        path, state, tensor_state = resume_checkpoint(output, resume, current_signature)
        del initial
        gc.collect()
        model = DiffusionModel.load(path / "model", device, trainable=True)
        model.configure(cfg.model, cfg.generation, cfg.scheduler)
        optimizer = build(model, cfg)
        optimizer.load_state_dict(tensor_state["optimizer"])
        restore_rng(tensor_state["rng"], device)
        baseline_test = json.loads((output / "baseline_test.json").read_text())
    else:
        model = initial
        del initial
        optimizer = build(model, cfg)
        save_export(model, output / "baseline")
        baseline_validation = evaluate(model, prepared, cfg, "validation")
        baseline_test = evaluate(model, prepared, cfg, "test")
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "environment.json", environment())
        write_json(output / "data_summary.json", summary)
        write_json(output / "baseline_validation.json", baseline_validation)
        write_json(output / "baseline_test.json", baseline_test)
        state = {
            "signature": current_signature,
            "best_epoch": 0,
            "best_metric": select_metric(
                baseline_validation["metrics"], cfg.evaluation.selection_metric
            )["value"],
            "history": [],
        }
        save_checkpoint(output, 0, model, optimizer, state)
        state["epoch"] = 0
    if cfg.training.gradient_checkpointing:
        model.pipeline.unet.enable_gradient_checkpointing()
    for epoch in range(state["epoch"] + 1, cfg.training.epochs + 1):
        training = train_epoch(model, prepared["train"], cfg, optimizer)
        validation = evaluate(model, prepared, cfg, "validation")
        selection = select_metric(validation["metrics"], cfg.evaluation.selection_metric)
        metric = selection["value"]
        if is_improvement(metric, state["best_metric"], selection["direction"]):
            state.update(best_metric=metric, best_epoch=epoch)
        state["history"].append({"epoch": epoch, "training": training, "validation": validation})
        state["epoch"] = epoch
        save_checkpoint(output, epoch, model, optimizer, state)
        for callback in callbacks:
            callback(epoch, state)
    save_export(model, output / "latest_export")
    del optimizer, model
    gc.collect()
    selected = DiffusionModel.load(
        output / "checkpoints" / f"checkpoint-{state['best_epoch']}" / "model", device
    )
    selected.configure(cfg.model, cfg.generation, cfg.scheduler)
    final = evaluate(selected, prepared, cfg, "test")
    save_export(selected, output / "export")
    sample_dir = output / "samples" / "selected-test"
    if not (sample_dir / "images.json").exists():
        selected.save_images(
            [r["caption"] for r in prepared["test"][: cfg.evaluation.sample_count]], sample_dir
        )
    record = {
        "schema_version": 1,
        "kind": "diffusion",
        "data": summary,
        "baseline_test": baseline_test,
        "finetuned_test": final,
        "export": str(output / "export"),
        "latest_export": str(output / "latest_export"),
        "samples": str(sample_dir / "images.json"),
        "device": str(device),
        "training": {
            "method": cfg.training.method,
            "trainable_components": ["unet"],
            "epochs": state["epoch"],
            "best_epoch": state["best_epoch"],
            "best_metric": state["best_metric"],
            "history": state["history"],
        },
        "note": "Validation denoising MSE selects including epoch zero; test is report-only. "
        "Fixed synthetic metrics do not establish generation quality.",
    }
    write_json(output / "report.json", record)
    return record
