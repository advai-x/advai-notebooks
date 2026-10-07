"""Independent grpo lifecycle: baseline, online RL, validation selection, test and export."""

import json

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.runtime import setup
from advai_notebooks.llm.rl_runtime import (
    resolve_checkpoint,
    restore_rng,
    run_signature,
    save_checkpoint,
    single_process,
)
from advai_notebooks.llm.runtime import environment

from .config import validate_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .metrics import selection_score
from .model import GRPOModel
from .optimizer import create_optimizer
from .train import train_epoch


def run(cfg, resume=None):
    validate_config(cfg)
    single_process()
    device = setup(cfg)
    rows, summary = load_data(cfg)
    model = GRPOModel(cfg.model, device, cfg.root, training=cfg.training, generation=cfg.generation)
    prepared = prepare_data(rows, model, cfg)
    fingerprint = run_signature(cfg, summary, model)
    output = cfg.path(cfg.output_dir)
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output is nonempty; resume latest or choose a fresh output_dir")
        output.mkdir(parents=True, exist_ok=True)
        reference = model.frozen_reference()
        optimizer = create_optimizer(model, cfg.training)
        baseline_validation = evaluate(model, prepared, cfg, "validation")
        baseline_test = evaluate(model, prepared, cfg, "test")
        state = {
            "signature": fingerprint,
            "history": [],
            "best_epoch": 0,
            "best_score": selection_score(baseline_validation),
            "baseline_validation": baseline_validation,
            "baseline_test": baseline_test,
        }
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "environment.json", environment(cfg, model))
        checkpoint = save_checkpoint(output, 0, model, reference, optimizer, state)
        start_epoch = 0
    else:
        checkpoint, state, tensors = resolve_checkpoint(output, resume, fingerprint)
        latest = json.loads((output / "latest.json").read_text())["checkpoint"]
        if checkpoint != (output / latest).resolve():
            raise ValueError("Resume latest only; use a separate output for branching")
        del model
        model = GRPOModel.load(checkpoint / "policy", device, trainable=True)
        reference = GRPOModel.load(checkpoint / "reference", device)
        reference.model.requires_grad_(False)
        model.configure(cfg.model, cfg.generation)
        prepared = prepare_data(rows, model, cfg)
        optimizer = create_optimizer(model, cfg.training)
        optimizer.load_state_dict(tensors["optimizer"])
        restore_rng(tensors["rng"])
        start_epoch = state["epoch"]
    for epoch in range(start_epoch + 1, cfg.training.epochs + 1):
        training = train_epoch(model, reference, prepared["train"], cfg, optimizer)
        validation = evaluate(model, prepared, cfg, "validation")
        score = selection_score(validation)
        if score >= state["best_score"]:
            state["best_score"], state["best_epoch"] = score, epoch
        state["history"].append({"epoch": epoch, "training": training, "validation": validation})
        checkpoint = save_checkpoint(output, epoch, model, reference, optimizer, state)
        print(
            json.dumps(
                {
                    "kind": cfg.kind,
                    "epoch": epoch,
                    "reward": score,
                    "optimizer_steps": training["optimizer_steps"],
                }
            ),
            flush=True,
        )
    latest_export = model.save(output / "latest_export")
    best = output / "checkpoints" / f"checkpoint-{state['best_epoch']}" / "policy"
    selected = GRPOModel.load(best, device)
    selected.configure(cfg.model, cfg.generation)
    export = selected.save(output / "export")
    report = {
        "schema_version": 1,
        "kind": cfg.kind,
        "data": summary,
        "baseline_test": state["baseline_test"],
        "finetuned_test": evaluate(selected, prepared, cfg, "test"),
        "export": str(export),
        "latest_export": str(latest_export),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "training": {
            "epochs": cfg.training.epochs,
            "best_epoch": state["best_epoch"],
            "history": state["history"],
        },
        "note": "Synthetic tiny runs validate the algorithm lifecycle, not model quality.",
    }
    write_json(output / "report.json", report)
    return report
