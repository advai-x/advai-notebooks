"""Independent instruct baseline/train/validation/test/export lifecycle."""

import gc
import json

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.runtime import setup
from advai_notebooks.llm.runtime import environment, resume_checkpoint, signature

from .data import load_data, prepare_data
from .evaluate import evaluate
from .model import InstructModel
from .train import train_model


def run(cfg, resume=None, *, callbacks=()):
    device = setup(cfg)
    rows, summary = load_data(cfg)
    output = cfg.path(cfg.output_dir)
    if resume is None and output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new output_dir or resume")
    resume_path, state = resume_checkpoint(output, resume) if resume else (None, None)
    model = (
        InstructModel.load(resume_path, device, trainable=True)
        if resume
        else InstructModel(cfg.model, device, cfg.root, cfg.training, cfg.generation)
    )
    prepared = prepare_data(rows, model, cfg)
    summary["encoded_rows"] = {name: len(values) for name, values in prepared.items()}
    summary["truncated_rows"] = sum(
        row.get("truncated", False) for split in prepared.values() for row in split
    )
    summary["dropped_tokens"] = sum(
        row.get("dropped_tokens", 0) for split in prepared.values() for row in split
    )
    current_signature = signature(cfg, summary, model)
    if state and state["signature"] != current_signature:
        raise ValueError("Resume configuration/data/tokenizer/implementation/runtime differs")
    output.mkdir(parents=True, exist_ok=True)
    if state is None:
        model.save(output / "baseline")
        baseline_validation = evaluate(model, prepared, cfg, "validation", generate_metrics=False)
        baseline_test = evaluate(model, prepared, cfg, "test", raw_rows=rows)
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "data_summary.json", summary)
        write_json(output / "environment.json", environment(cfg, model))
        write_json(output / "baseline_validation.json", baseline_validation)
        write_json(output / "baseline_test.json", baseline_test)
    else:
        baseline_validation = json.loads((output / "baseline_validation.json").read_text())
        baseline_test = json.loads((output / "baseline_test.json").read_text())
    state = train_model(
        model, prepared, cfg, current_signature, baseline_validation, resume_path, state, callbacks
    )
    # Release the training model before allocating selected weights on the device.
    del model
    gc.collect()
    selected = InstructModel.load(output / state["best_checkpoint"], device)
    selected.save(output / "best")
    selected.save(output / "export")
    final = evaluate(selected, prepared, cfg, "test", raw_rows=rows)
    record = {
        "kind": cfg.kind,
        "data": summary,
        "baseline_test": baseline_test,
        "finetuned_test": final,
        "export": str(output / "export"),
        "device": str(device),
        "training": {**state, "completed": state["last_epoch"] >= cfg.training.epochs},
        "note": "Validation nll selects checkpoints including epoch zero; test is report-only.",
    }
    write_json(output / "report.json", record)
    return record
