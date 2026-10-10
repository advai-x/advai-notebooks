"""Held-out teacher-forcing loss; this is NOT a perceptual speech-quality score."""

import torch

from .losses import loss


def evaluate(model, prepared, cfg, split):
    model.model.eval()
    rows = prepared[split]
    with torch.inference_mode():
        values = [float(loss(model, [row])) for row in rows]
    return {
        "kind": "tts",
        "split": split,
        "n": len(rows),
        "metrics": {"loss": sum(values) / len(values)},
        "predictions": [{"id": row["id"], "loss": value} for row, value in zip(rows, values)],
        "note": "Teacher-forcing loss selects checkpoints; fixed-text listening is required to judge speech quality.",
    }
