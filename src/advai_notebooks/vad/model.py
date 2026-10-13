"""Frozen Silero features with an eager, compatible recurrent decoder.

Decoder architecture follows Silero VAD's MIT-licensed tuning interface:
https://github.com/snakers4/silero-vad/blob/master/tuning/utils.py
This wrapper owns batching, persistence and bounded offline inference.
"""

import hashlib
import importlib.metadata
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.audio import read_audio, write_audio
from advai_notebooks.common.hub import resolve_export

from .config import ModelConfig, validate_model
from .notice import SILERO_LICENSE
from .segments import speech_intervals


class RecurrentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(nn.Dropout(0.1), nn.ReLU(), nn.Conv1d(128, 1, 1), nn.Sigmoid())

    def forward(self, features):
        state, predictions = None, []
        for frame in features.unbind(0):
            hidden, cell = self.rnn(frame.unsqueeze(0), state)
            state = (hidden, cell)
            predictions.append(self.decoder(hidden.unsqueeze(-1)).reshape(()))
        return torch.stack(predictions)


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 128)

    def forward(self, frames):
        stats = torch.stack(
            (
                frames.mean(-1),
                frames.abs().mean(-1),
                frames.square().mean(-1),
                frames.abs().amax(-1),
            ),
            dim=-1,
        )
        return torch.tanh(self.projection(stats))


class Network(nn.Module):
    def __init__(self, encoder, decoder, stft=None):
        super().__init__()
        self.encoder, self.decoder, self.stft = encoder, decoder, stft
        for module in (encoder, stft):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if self.stft is not None:
            self.stft.eval()
        return self


class VADModel:
    def __init__(self, settings, device="cpu", root=None, training=None, *, native_bytes=None):
        validate_model(settings)
        self.settings, self.device = settings, torch.device(device)
        self.native_bytes = native_bytes
        if settings.id == "silero":
            if native_bytes is None:
                package = importlib.metadata.distribution("silero-vad")
                if package.version != settings.revision:
                    raise ValueError("Installed Silero package differs from pinned revision")
                native_bytes = package.locate_file("silero_vad/data/silero_vad.jit").read_bytes()
                if hashlib.sha256(native_bytes).hexdigest() != settings.sha256:
                    raise ValueError("Bundled Silero model sha256 differs from pinned asset")
                self.native_bytes = native_bytes
            native = torch.jit.load(io.BytesIO(native_bytes), map_location="cpu").eval()
            branch = native._model if settings.sample_rate == 16000 else native._model_8k
            decoder = RecurrentDecoder()
            decoder.load_state_dict(branch.decoder.state_dict(), strict=True)
            self.model = Network(branch.encoder, decoder, branch.stft)
        else:
            self.model = Network(TinyFeatures(), RecurrentDecoder())
        self.model.to(self.device).eval()

    @torch.no_grad()
    def features(self, waveform):
        samples = torch.as_tensor(np.asarray(waveform), dtype=torch.float32, device=self.device)
        width = self.settings.sample_rate * 32 // 1000
        padded = nn.functional.pad(samples, (0, (-len(samples)) % width))
        self.model.encoder.eval()
        if self.settings.id == "tiny":
            return self.model.encoder(padded.reshape(-1, width)).cpu()
        context = width // 8
        padded = nn.functional.pad(padded, (context, 0))
        features = []
        # Small batches bound STFT memory; context comes from the preceding raw samples.
        windows = padded.unfold(0, width + context, width)
        for batch in windows.split(128):
            features.append(self.model.encoder(self.model.stft(batch.contiguous())).squeeze(-1))
        return torch.cat(features).cpu()

    def probabilities(self, features):
        return self.model.decoder(features.to(self.device))

    def loss_weight(self, batch):
        return sum(float(row["weights"].sum()) for row in batch)

    @torch.no_grad()
    def detect(self, audio):
        waveform, _ = read_audio(
            audio,
            self.settings.sample_rate,
            min_duration=0.001,
            max_duration=self.settings.max_inference_duration,
        )
        self.model.eval()
        probabilities = self.probabilities(self.features(waveform)).cpu().tolist()
        duration = len(waveform) / self.settings.sample_rate
        return {
            "audio": str(audio),
            "sample_rate": self.settings.sample_rate,
            "duration": duration,
            "frame_seconds": 0.032,
            "probabilities": probabilities,
            "speech_segments": speech_intervals(probabilities, duration, self.settings),
        }

    def segment(self, audio, output_dir, *, max_chunk_duration=30.0):
        if isinstance(max_chunk_duration, bool) or not isinstance(max_chunk_duration, (int, float)):
            raise ValueError("max_chunk_duration must be a finite positive number")
        if (
            not np.isfinite(max_chunk_duration)
            or max_chunk_duration < 1 / self.settings.sample_rate
        ):
            raise ValueError("max_chunk_duration must be finite and at least one audio sample")
        directory = Path(output_dir)
        if directory.exists():
            raise ValueError("Segment destination must be new; existing files are never replaced")
        result = self.detect(audio)
        waveform, _ = read_audio(
            audio,
            self.settings.sample_rate,
            min_duration=0.001,
            max_duration=self.settings.max_inference_duration,
        )
        directory.mkdir(parents=True)
        width = max(1, int(max_chunk_duration * self.settings.sample_rate))
        chunks = []
        for interval in result["speech_segments"]:
            first = max(0, round(interval["start"] * self.settings.sample_rate))
            last = min(len(waveform), round(interval["end"] * self.settings.sample_rate))
            for start in range(first, last, width):
                end = min(start + width, last)
                metadata = write_audio(
                    directory / f"chunk-{len(chunks):04d}.wav",
                    waveform[start:end],
                    self.settings.sample_rate,
                )
                chunks.append(
                    {
                        **metadata,
                        "source_start": start / self.settings.sample_rate,
                        "source_end": end / self.settings.sample_rate,
                    }
                )
        write_json(
            directory / "segments.json",
            {
                **result,
                "chunks": chunks,
                "note": "No transcripts. Align labels before using chunks for STT training.",
            },
        )
        return {**result, "chunks": chunks, "manifest": str(directory / "segments.json")}

    def save(self, path):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if self.settings.id == "silero":
            native = torch.jit.load(io.BytesIO(self.native_bytes), map_location="cpu").eval()
            branch = native._model if self.settings.sample_rate == 16000 else native._model_8k
            state = {
                name: value.detach().cpu()
                for name, value in self.model.decoder.state_dict().items()
            }
            branch.decoder.load_state_dict(state, strict=True)
            native.reset_states()
            torch.jit.save(native, str(destination / "silero_vad.jit"))
            (destination / "LICENSE").write_text(SILERO_LICENSE, encoding="utf-8")
            asset = "silero_vad.jit"
        else:
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()},
                destination / "model.safetensors",
            )
            asset = "model.safetensors"
        write_json(
            destination / "advai_metadata.json",
            {
                "kind": "vad",
                "schema_version": 1,
                "model": asdict(self.settings),
                "asset": asset,
                "asset_sha256": hashlib.sha256((destination / asset).read_bytes()).hexdigest(),
                "method": "decoder",
                "note": "Synthetic smoke is not evidence of speech detection quality.",
            },
        )
        return destination

    @classmethod
    def load(cls, path, device="cpu", trainable=False, **hub_options):
        directory = resolve_export(path, **hub_options)
        metadata = json.loads((directory / "advai_metadata.json").read_text())
        if metadata.get("kind") != "vad" or metadata.get("schema_version") != 1:
            raise ValueError("Expected a VAD inference export")
        settings = ModelConfig(**metadata["model"])
        expected = "silero_vad.jit" if settings.id == "silero" else "model.safetensors"
        if metadata["asset"] != expected:
            raise ValueError("Unexpected VAD asset name")
        contents = (directory / expected).read_bytes()
        if hashlib.sha256(contents).hexdigest() != metadata["asset_sha256"]:
            raise ValueError("VAD inference asset checksum mismatch")
        if settings.id == "silero" and (
            not (directory / "LICENSE").is_file()
            or (directory / "LICENSE").read_text(encoding="utf-8") != SILERO_LICENSE
        ):
            raise ValueError("Native Silero exports must retain the upstream MIT license")
        result = cls(settings, device, native_bytes=contents if settings.id == "silero" else None)
        if settings.id == "tiny":
            result.model.load_state_dict(load_file(directory / expected, device=str(device)))
        result.model.train(trainable)
        return result
