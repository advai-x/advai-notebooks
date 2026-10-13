"""Corpus confusion counts and interval-duration scoring; no tolerance collar."""

import numpy as np


def rates(tp, fp, fn, tn):
    def ratio(a, b):
        return float(a / b) if b else 0.0

    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "miss_rate": ratio(fn, tp + fn),
        "false_alarm_rate": ratio(fp, fp + tn),
        "error_rate": ratio(fp + fn, tp + fp + fn + tn),
    }


def frame_counts(labels, probabilities, weights, threshold):
    truth, detected = np.asarray(labels) > 0.5, np.asarray(probabilities) >= threshold
    seconds = np.asarray(weights) * 0.032
    return np.asarray(
        [
            seconds[truth & detected].sum(),
            seconds[~truth & detected].sum(),
            seconds[truth & ~detected].sum(),
            seconds[~truth & ~detected].sum(),
        ]
    )


def interval_counts(reference, predicted, duration):
    tp = sum(
        max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
        for a in reference
        for b in predicted
    )
    speech = sum(s["end"] - s["start"] for s in reference)
    positive = sum(s["end"] - s["start"] for s in predicted)
    fp, fn = max(0, positive - tp), max(0, speech - tp)
    return np.asarray([tp, fp, fn, max(0, duration - tp - fp - fn)])
