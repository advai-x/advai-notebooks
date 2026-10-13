"""Frame loss and postprocessed interval metrics, with explicit denominators."""

import numpy as np
import torch

from .losses import loss
from .metrics import frame_counts, interval_counts, rates
from .segments import speech_intervals


@torch.no_grad()
def predictions(model, rows):
    model.model.eval()
    return [
        {**row, "probabilities": model.probabilities(row["features"]).cpu().numpy()} for row in rows
    ]


def score(rows, settings):
    frames, intervals = np.zeros(4), np.zeros(4)
    items = []
    for row in rows:
        detected = speech_intervals(row["probabilities"], row["duration"], settings)
        frames += frame_counts(
            row["labels"], row["probabilities"], row["weights"], settings.threshold
        )
        intervals += interval_counts(row["speech_segments"], detected, row["duration"])
        items.append({"id": row["id"], "reference": row["speech_segments"], "predicted": detected})
    return {
        "metrics": {**rates(*frames), **{f"segment_{k}": v for k, v in rates(*intervals).items()}},
        "frame_seconds": dict(zip(("tp", "fp", "fn", "tn"), frames.tolist())),
        "segment_seconds": dict(zip(("tp", "fp", "fn", "tn"), intervals.tolist())),
        "threshold": settings.threshold,
        "exit_threshold": settings.exit_threshold,
        "items": items,
        "note": "Frame labels use 32ms majority occupancy; interval scores have no collar.",
    }


@torch.no_grad()
def evaluate(model, prepared, cfg, split):
    rows = prepared[split]
    model.model.eval()
    result = score(predictions(model, rows), model.settings)
    result["metrics"]["bce"] = float(loss(model, rows))
    return {"split": split, **result}
