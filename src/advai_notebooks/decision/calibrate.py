"""Bounded temperature fitting on calibration data with sparse-group fallback."""

import math
from collections import Counter


def temp_bucket(qtype, options):
    return f"{qtype}:{options}"


def _fit_temperature(rows):
    import torch

    width = max(len(row["logits"]) for row in rows)
    logits = torch.full((len(rows), width), -1e4, dtype=torch.float64)
    targets = torch.zeros_like(logits)
    for index, row in enumerate(rows):
        size = len(row["logits"])
        logits[index, :size] = torch.tensor(row["logits"], dtype=torch.float64)
        targets[index, :size] = torch.tensor(row["target"], dtype=torch.float64)

    def nll(log_temperature):
        return float(
            -(targets * torch.log_softmax(logits / math.exp(log_temperature), -1)).sum(-1).mean()
        )

    # Scalar golden-section search is deterministic and needs no optional optimizer/scipy.
    left, right = -4.0, 4.0
    ratio = (math.sqrt(5) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    fa, fb = nll(a), nll(b)
    for _ in range(64):
        if fa < fb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = nll(a)
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = nll(b)
    # Identity is always a candidate, so fitted calibration NLL cannot regress.
    selected = min((0.0, (left + right) / 2, -4.0, 4.0), key=nll)
    return math.exp(selected)


def fit_calibration(predictions, config=None):
    from .config import CalibrationConfig

    config = config or CalibrationConfig()
    if not predictions:
        raise ValueError("Cannot fit calibration on empty predictions")
    for row in predictions:
        p, t = row["logits"], row["target"]
        if (
            row["qtype"] not in range(3)
            or len(p) < 2
            or len(p) != len(t)
            or any(not math.isfinite(v) for v in p)
            or any(not math.isfinite(v) or v < 0 for v in t)
            or not math.isclose(sum(t), 1, abs_tol=1e-6)
        ):
            raise ValueError("Invalid calibration logits or targets")
    counts = Counter(row["qtype"] for row in predictions)
    buckets = Counter(temp_bucket(row["qtype"], len(row["target"])) for row in predictions)
    temperatures = [1.0] * 3
    for kind in range(3):
        if counts[kind] >= config.min_type_samples:
            temperatures[kind] = _fit_temperature(
                [row for row in predictions if row["qtype"] == kind]
            )
    by_options = {}
    for bucket, count in buckets.items():
        if count >= config.min_bucket_samples:
            by_options[bucket] = _fit_temperature(
                [
                    row
                    for row in predictions
                    if temp_bucket(row["qtype"], len(row["target"])) == bucket
                ]
            )
    return {
        "temperature": temperatures,
        "temperature_by_options": by_options,
        "samples_by_type": {str(k): counts[k] for k in range(3)},
        "samples_by_bucket": dict(buckets),
        "type_min_samples": config.min_type_samples,
        "bucket_min_samples": config.min_bucket_samples,
        "unfitted_types": [k for k in range(3) if counts[k] < config.min_type_samples],
        "split": "calibration",
        "method": "bounded_nll_temperature",
    }
