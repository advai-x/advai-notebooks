"""Deterministic nonspeech/tone VAD mechanics fixtures; never overwrite differing audio."""

from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("one", "two", "three", "four", "five", "six", "seven", "eight")


def prepare(root=ROOT):
    rate = 16000
    for index, name in enumerate(NAMES):
        path = root / "data/sample/vad/audio" / f"{name}.wav"
        time = np.arange(rate * 2, dtype=np.float64) / rate
        signal = 0.001 * np.cos(2 * np.pi * (130 + 11 * index) * time)
        if index not in {2, 5}:
            active = ((time >= 0.32) & (time < 0.8)) | ((time >= 1.12) & (time < 1.6))
            signal[active] += 0.2 * np.sin(2 * np.pi * (220 + index * 37) * time[active])
        signal = signal.astype(np.float32)
        # Canonicalize trigonometric zero crossings before explicit PCM16 quantization.
        # Avoid platform-dependent float-to-integer conversion inside libsndfile.
        signal[np.abs(signal) < 1e-10] = 0
        pcm = np.floor(signal * 32768).astype(np.int16)
        if path.exists():
            existing, actual_rate = sf.read(path, dtype="int16")
            if actual_rate != rate or not np.array_equal(existing, pcm):
                raise ValueError(f"Refusing to overwrite differing fixture {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, pcm, rate, subtype="PCM_16")


if __name__ == "__main__":
    prepare()
