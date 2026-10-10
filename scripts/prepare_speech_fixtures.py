"""Regenerate only the sixteen named synthetic PCM fixtures; never read human speech."""

from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("one", "two", "three", "four", "five", "six", "seven", "eight")


def prepare(root=ROOT):
    for kind, rate in (("stt", 16000), ("tts", 24000)):
        for i, name in enumerate(NAMES):
            path = root / "data/sample" / kind / "audio" / f"{name}.wav"
            time = np.arange(rate // 4, dtype=np.float64) / rate
            signal = (0.15 * np.sin(2 * np.pi * (220 + i * 37) * time)).astype("float32")
            if path.exists():
                existing, actual_rate = sf.read(path, dtype="float32")
                expected = np.floor(signal * 32768).astype(np.int16).astype(np.float32) / 32768
                if actual_rate != rate or not np.array_equal(existing, expected):
                    raise ValueError(f"Existing fixture differs; refusing to overwrite {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, signal, rate, subtype="PCM_16")


if __name__ == "__main__":
    prepare()
