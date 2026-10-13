"""Offline hysteresis, silence merging and padded interval construction."""

import math

import numpy as np


def validate_segments(value, duration):
    if not isinstance(value, list):
        raise ValueError("speech_segments must be a list; [] explicitly means no speech")
    previous = 0.0
    for item in value:
        if not isinstance(item, dict) or set(item) != {"start", "end"}:
            raise ValueError("Each speech segment requires only start/end in seconds")
        start, end = item["start"], item["end"]
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (start, end)
        ):
            raise ValueError("Segment boundaries must be finite seconds")
        if not previous <= start < end <= duration + 1e-7:
            raise ValueError("Speech segments must be ordered, nonoverlapping and within audio")
        previous = end
    return value


def labels_for_frames(segments, samples, sample_rate):
    """Majority occupancy per 32 ms frame; padding never counts as supervision."""
    width = sample_rate * 32 // 1000
    targets, weights = [], []
    for offset in range(0, samples, width):
        end = min(offset + width, samples)
        start_seconds, end_seconds = offset / sample_rate, end / sample_rate
        occupied = sum(
            max(0, min(end_seconds, s["end"]) - max(start_seconds, s["start"])) for s in segments
        )
        targets.append(float(occupied > (end_seconds - start_seconds) / 2 + 1e-10))
        weights.append((end - offset) / width)
    return np.asarray(targets, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def speech_intervals(probabilities, duration, settings):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Frame probabilities must be nonempty and finite")
    if np.any((values < 0) | (values > 1)) or duration <= 0:
        raise ValueError("Invalid probabilities or audio duration")
    if len(values) != math.ceil(
        round(duration * settings.sample_rate) / (settings.sample_rate * 32 // 1000)
    ):
        raise ValueError("Probability count must match the complete audio, including final frame")
    raw, start, silence = [], None, None
    for index, value in enumerate(values):
        time = index * 0.032
        if start is None:
            if value >= settings.threshold:
                start = time
            continue
        if value >= settings.threshold:
            silence = None
        elif value < settings.exit_threshold:
            if silence is None:
                silence = time
            if (index + 1) * 0.032 - silence + 1e-10 >= settings.min_silence_duration:
                raw.append((start, silence))
                start, silence = None, None
        # Values between thresholds preserve the current state and pending silence.
    if start is not None:
        raw.append((start, min(duration, silence) if silence is not None else duration))
    intervals = []
    for start, end in raw:
        if end - start + 1e-10 < settings.min_speech_duration:
            continue
        start, end = max(0, start - settings.speech_pad), min(duration, end + settings.speech_pad)
        if intervals and start <= intervals[-1]["end"]:
            intervals[-1]["end"] = max(intervals[-1]["end"], end)
        elif end > start:
            intervals.append({"start": start, "end": end})
    return intervals
