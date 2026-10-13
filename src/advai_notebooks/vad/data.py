"""Local audio and interval annotations with source-group leakage checks."""

from advai_notebooks.common.artifacts import digest
from advai_notebooks.common.audio import local_audio
from advai_notebooks.common.media import records, split_rows

from .segments import labels_for_frames, validate_segments


def load_data(cfg):
    manifest = cfg.path(cfg.data.samples)
    rows, summary = records(manifest)
    seen, hashes = {}, []
    for row in rows:
        path, waveform, fingerprints = local_audio(
            manifest.parent,
            row.get("audio"),
            cfg.model.sample_rate,
            min_duration=cfg.data.min_duration,
            max_duration=cfg.data.max_duration,
        )
        duration = len(waveform) / cfg.model.sample_rate
        validate_segments(row.get("speech_segments"), duration)
        for fingerprint in fingerprints:
            if seen.setdefault(fingerprint, row["split"]) != row["split"]:
                raise ValueError("Duplicate decoded/resampled audio crosses splits")
        hashes.append(fingerprints)
        row.update(audio=str(path), waveform=waveform, duration=duration)
    for split, split_items in split_rows(rows).items():
        labels = [
            labels_for_frames(r["speech_segments"], len(r["waveform"]), cfg.model.sample_rate)[0]
            for r in split_items
        ]
        if not any((y == 1).any() for y in labels) or not any((y == 0).any() for y in labels):
            raise ValueError(f"{split} must contain both speech and nonspeech frames")
    summary.update(sample_rate=cfg.model.sample_rate, media_fingerprint=digest(hashes))
    return rows, summary


def prepare_data(rows, model, cfg, fit=False):
    if model.settings.sample_rate != cfg.model.sample_rate:
        raise ValueError("Export and dataset sample rates differ")
    prepared = []
    for row in rows:
        labels, weights = labels_for_frames(
            row["speech_segments"], len(row["waveform"]), cfg.model.sample_rate
        )
        prepared.append(
            {
                **row,
                "features": model.features(row["waveform"]),
                "labels": labels,
                "weights": weights,
            }
        )
    return split_rows(prepared)
