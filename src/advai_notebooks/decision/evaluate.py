"""Decision evaluation orchestration: predictions, metrics and optional calibration."""

from .calibrate import fit_calibration
from .metrics import evaluate_predictions


def predict_items(bundle, items, device, batch_size, precision="fp32", calibrated=True):
    """Compatibility wrapper; numerical inference lives in the model facade."""
    return bundle.model.to(device).predict_items(items, batch_size, precision, calibrated)


def evaluate(model, data, cfg, split, calibrated=True):
    """Return (metrics, predictions), matching embedding/ranker evaluator boundaries."""
    predictions = model.predict_items(
        data[split],
        cfg.training.batch_size,
        cfg.runtime.precision,
        calibrated,
    )
    return evaluate_predictions(predictions), predictions


def fit_postprocess(model, data, cfg):
    """Fit using the calibration split only; disabled calibration is identity."""
    model.config["temperature"] = [1.0] * 3
    model.config.pop("temperature_by_options", None)
    if not cfg.calibration.enabled:
        return None
    predictions = model.predict_items(
        data["calibration"], cfg.training.batch_size, cfg.runtime.precision, calibrated=False
    )
    calibration = fit_calibration(predictions, cfg.calibration)
    model.config.update(
        {
            "temperature": calibration["temperature"],
            "temperature_by_options": calibration["temperature_by_options"],
        }
    )
    return calibration


def evaluate_selected(model, data, cfg):
    """Evaluate raw/fitted selected weights without using test labels for fitting."""
    raw_metrics, _ = evaluate(model, data, cfg, "test", calibrated=False)
    calibration = fit_postprocess(model, data, cfg)
    metrics, predictions = evaluate(model, data, cfg, "test")
    return {
        "fine_tuned": raw_metrics,
        "fine_tuned_calibrated": metrics if calibration else None,
        "calibration": calibration,
        "predictions": predictions,
    }
