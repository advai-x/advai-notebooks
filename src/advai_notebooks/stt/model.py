"""Native Whisper initialization, feature/label batching, transcription and persistence."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.audio import read_audio
from advai_notebooks.common.hub import resolve_export
from advai_notebooks.common.media import text

from .config import ModelConfig


class STTModel:
    def __init__(self, settings=None, device="cpu", root=None, training=None):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.settings, self.device = settings or ModelConfig(), torch.device(device)
        if self.settings.id == "tiny":
            self.model, self.processor = self._tiny()
        else:
            local = Path(root or ".") / self.settings.id
            if not local.is_dir() and not self.settings.revision:
                raise ValueError("Pin model.revision when loading a Hub base")
            path = str(local.resolve()) if local.is_dir() else self.settings.id
            options = {"revision": self.settings.revision, "trust_remote_code": False}
            self.model = WhisperForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.float32, **options
            )
            self.processor = WhisperProcessor.from_pretrained(
                path, language=self.settings.language, task="transcribe", **options
            )
        self._configure()

    def _tiny(self):
        from transformers import (
            WhisperConfig,
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
        )

        from advai_notebooks.common.toy_processor import tokenizer

        tokens = tokenizer()

        # A small native Whisper; its synthetic tokenizer is not a pretrained Whisper tokenizer.
        class Processor:
            pass

        processor = Processor()
        processor.tokenizer = tokens
        processor.feature_extractor = WhisperFeatureExtractor(feature_size=16, chunk_length=1)
        config = WhisperConfig(
            vocab_size=len(tokens),
            num_mel_bins=16,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_source_positions=50,
            max_target_positions=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
            suppress_tokens=[],
            begin_suppress_tokens=[],
        )
        return WhisperForConditionalGeneration(config), processor

    def _configure(self):
        if self.model.config.model_type != "whisper":
            raise ValueError("STT backend requires a native Whisper model")
        if self.settings.max_label_tokens > self.model.config.max_target_positions:
            raise ValueError("max_label_tokens exceeds the native decoder context")
        prefix_length = 1 if self.settings.id == "tiny" else 4
        if self.settings.max_new_tokens + prefix_length > self.model.config.max_target_positions:
            raise ValueError("max_new_tokens exceeds the native decoder context")
        self.model.config.use_cache = False
        self.model.generation_config.forced_decoder_ids = None
        if self.settings.id != "tiny":
            self.model.generation_config.language = self.settings.language
            self.model.generation_config.task = "transcribe"
        self.model.to(device=self.device, dtype=torch.float32)

    def encode(self, row):
        features = self.processor.feature_extractor(
            row["waveform"], sampling_rate=16000, return_tensors="pt"
        )["input_features"][0]
        labels = self.processor.tokenizer(row["text"], return_tensors="pt")["input_ids"][0]
        if self.settings.id == "tiny":
            labels = torch.cat((labels, torch.tensor([self.model.config.eos_token_id])))
        start = self.model.config.decoder_start_token_id
        if len(labels) and labels[0] == start:
            labels = labels[1:]
        if not len(labels) or len(labels) > self.settings.max_label_tokens:
            raise ValueError("Transcript exceeds label budget; no silent truncation")
        return {"input_features": features, "labels": labels, "text": row["text"], "id": row["id"]}

    def batch(self, rows):
        length = max(len(row["labels"]) for row in rows)
        labels = torch.stack(
            [
                torch.nn.functional.pad(row["labels"], (0, length - len(row["labels"])), value=-100)
                for row in rows
            ]
        )
        return {
            "input_features": torch.stack([row["input_features"] for row in rows]).to(self.device),
            "labels": labels.to(self.device),
        }

    def loss_weight(self, rows):
        return sum(len(row["labels"]) for row in rows)

    def decode_features(self, features):
        self.model.eval()
        with torch.inference_mode():
            ids = self.model.generate(
                features.unsqueeze(0).to(self.device),
                do_sample=False,
                max_new_tokens=self.settings.max_new_tokens,
                use_cache=True,
            )
        return self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)

    def transcribe(self, audio):
        text(str(audio), "audio")
        waveform, _ = read_audio(
            audio, 16000, max_duration=1.0 if self.settings.id == "tiny" else 30.0
        )
        features = self.processor.feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )["input_features"][0]
        return self.decode_features(features)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.tokenizer.save_pretrained(path)
        self.processor.feature_extractor.save_pretrained(path)
        write_json(
            path / "advai_metadata.json",
            {"schema_version": 1, "kind": "stt", "model": asdict(self.settings)},
        )
        return path

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **options):
        from transformers import (
            PreTrainedTokenizerFast,
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        path = resolve_export(path, **options)
        meta = json.loads((path / "advai_metadata.json").read_text())
        if meta.get("kind") != "stt" or meta.get("schema_version") != 1:
            raise ValueError("Not a supported STT export")
        obj = cls.__new__(cls)
        obj.settings, obj.device = ModelConfig(**meta["model"]), torch.device(device)
        obj.model = WhisperForConditionalGeneration.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.float32
        )
        if obj.settings.id == "tiny":

            class Processor:
                pass

            obj.processor = Processor()
            obj.processor.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                path, local_files_only=True
            )
            obj.processor.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                path, local_files_only=True
            )
        else:
            obj.processor = WhisperProcessor.from_pretrained(
                path, local_files_only=True, language=obj.settings.language, task="transcribe"
            )
        obj._configure()
        return obj
