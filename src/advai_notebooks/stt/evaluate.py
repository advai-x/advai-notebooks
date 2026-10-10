"""Teacher-forced NLL and independent autoregressive held-out transcription errors."""

import torch

from .losses import loss
from .metrics import error_rates


def evaluate(model, prepared, cfg, split):
    model.model.eval()
    rows, total, tokens, predictions = prepared[split], 0.0, 0, []
    with torch.inference_mode():
        for row in rows:
            weight = model.loss_weight([row])
            total += float(loss(model, [row])) * weight
            tokens += weight
            predictions.append(
                {
                    "id": row["id"],
                    "reference": row["text"],
                    "prediction": model.decode_features(row["input_features"]),
                }
            )
    errors = error_rates(
        [r["reference"] for r in predictions],
        [r["prediction"] for r in predictions],
        cfg.evaluation.normalization,
    )
    return {
        "kind": "stt",
        "split": split,
        "n": len(rows),
        "metrics": {"nll": total / tokens, "wer": errors["wer"], "cer": errors["cer"]},
        "error_counts": errors,
        "normalization": cfg.evaluation.normalization,
        "predictions": predictions,
    }
