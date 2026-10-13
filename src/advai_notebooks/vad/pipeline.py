"""Epoch recovery followed by validation-only offline threshold calibration."""

from dataclasses import replace

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.native_training import run as execute
from advai_notebooks.common.runtime import resolve_device

from .config import validate_config
from .data import load_data, prepare_data
from .evaluate import evaluate, predictions, score
from .losses import loss
from .model import VADModel


def calibrate(model, prepared, cfg):
    validation = predictions(model, prepared["validation"])
    candidates = []
    for threshold in sorted(set(cfg.evaluation.thresholds)):
        settings = replace(
            model.settings,
            threshold=threshold,
            exit_threshold=max(0, threshold - cfg.evaluation.hysteresis),
        )
        metrics = score(validation, settings)["metrics"]
        candidates.append(
            {
                "threshold": threshold,
                "exit_threshold": settings.exit_threshold,
                "segment_f1": metrics["segment_f1"],
            }
        )
    # Deterministic tie break prefers the higher threshold (fewer false triggers).
    selected = max(candidates, key=lambda x: (x["segment_f1"], x["threshold"]))
    model.settings = replace(
        model.settings, threshold=selected["threshold"], exit_threshold=selected["exit_threshold"]
    )
    return {
        "split": "validation",
        "objective": "segment_f1",
        "candidates": candidates,
        "selected": selected,
        "note": "Test labels do not participate in calibration.",
    }


def run(cfg, resume=None, *, callbacks=()):
    validate_config(cfg)
    report = execute(
        cfg, VADModel, load_data, prepare_data, loss, evaluate, resume=resume, callbacks=callbacks
    )
    if cfg.evaluation.calibrate:
        device = resolve_device(cfg.runtime.device)
        selected = VADModel.load(report["export"], device)
        rows, _ = load_data(cfg)
        prepared = prepare_data(rows, selected, cfg)
        calibration = calibrate(selected, prepared, cfg)
        selected.save(report["export"])
        baseline = VADModel.load(cfg.path(cfg.output_dir) / "baseline", device)
        baseline.settings = replace(
            baseline.settings,
            threshold=selected.settings.threshold,
            exit_threshold=selected.settings.exit_threshold,
        )
        report.update(
            calibration=calibration,
            baseline_test=evaluate(baseline, prepared, cfg, "test"),
            finetuned_test=evaluate(selected, prepared, cfg, "test"),
        )
        write_json(cfg.path(cfg.output_dir) / "calibration.json", calibration)
        write_json(cfg.path(cfg.output_dir) / "report.json", report)
    return report
