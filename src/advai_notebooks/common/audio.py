"""Local PCM audio validation, resampling and content fingerprints; no task objectives."""

import hashlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .artifacts import digest
from .media import records, text


def read_audio(path, sample_rate, *, max_duration=30.0, min_duration=0.05):
    path = Path(path)
    info = sf.info(path)
    if info.channels != 1:
        raise ValueError("Audio must be mono; convert stereo deliberately before training")
    duration = info.frames / info.samplerate
    if not min_duration <= duration <= max_duration:
        raise ValueError("Audio duration is outside the configured bounds; no silent truncation")
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    if not len(waveform) or not np.isfinite(waveform).all() or np.max(np.abs(waveform)) > 1:
        raise ValueError("Audio must contain finite normalized PCM samples")
    original = hashlib.sha256(str(rate).encode() + waveform.tobytes()).hexdigest()
    if rate != sample_rate:
        divisor = gcd(rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, rate // divisor).astype(
            "float32"
        )
    canonical = hashlib.sha256(str(sample_rate).encode() + waveform.tobytes()).hexdigest()
    return waveform, (original, canonical)


def local_audio(root, value, sample_rate, **bounds):
    text(value, "audio")
    root, relative = Path(root).resolve(), Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Audio references must be relative to the dataset")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping audio reference")
    waveform, checksums = read_audio(path, sample_rate, **bounds)
    return path, waveform, checksums


def load_audio_records(cfg, sample_rate):
    path = cfg.path(cfg.data.samples)
    rows, summary = records(path)
    seen, checksums = {}, []
    for row in rows:
        text(row.get("text"), "text")
        audio, waveform, hashes = local_audio(
            path.parent,
            row.get("audio"),
            sample_rate,
            max_duration=cfg.data.max_duration,
            min_duration=cfg.data.min_duration,
        )
        for checksum in hashes:
            if seen.setdefault(checksum, row["split"]) != row["split"]:
                raise ValueError("Duplicate decoded/resampled audio crosses splits")
        checksums.append(hashes)
        row.update(audio=str(audio), waveform=waveform, audio_fingerprint=hashes[1])
    summary.update(media_fingerprint=digest(checksums), sample_rate=sample_rate)
    return rows, summary


def write_audio(path, waveform, sample_rate):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Generated audio must be a nonempty finite mono waveform")
    if np.max(np.abs(waveform)) > 1:
        raise ValueError("Generated audio exceeds normalized PCM range; no silent clipping")
    path = Path(path)
    if path.exists():
        raise ValueError("Audio output already exists; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate, subtype="PCM_16")
    return {"audio": str(path), "sample_rate": sample_rate, "duration": len(waveform) / sample_rate}
