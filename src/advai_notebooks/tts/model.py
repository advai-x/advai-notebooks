"""Qwen3-TTS Base adaptation and an explicitly synthetic offline smoke backend."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.audio import write_audio
from advai_notebooks.common.hub import resolve_export
from advai_notebooks.common.media import text

from .config import ModelConfig


class TinyTTS(torch.nn.Module):
    """Text-conditioned GRU over quantized PCM frames; not a pretrained speech model."""

    def __init__(self):
        super().__init__()
        self.text_embedding = torch.nn.Embedding(256, 16)
        self.code_embedding = torch.nn.Embedding(33, 16)
        self.decoder = torch.nn.GRU(16, 16, batch_first=True)
        self.output = torch.nn.Linear(16, 32)

    def forward(self, codes, text_ids):
        condition = self.text_embedding(text_ids).mean(1).unsqueeze(0)
        hidden, _ = self.decoder(self.code_embedding(codes), condition)
        return self.output(hidden)


class TTSModel:
    def __init__(self, settings=None, device="cpu", root=None, training=None):
        self.settings, self.device = settings or ModelConfig(), torch.device(device)
        self.voice = None
        self.base_path = None
        if self.settings.id == "tiny":
            self.model = TinyTTS().to(self.device)
        else:
            from qwen_tts import Qwen3TTSModel

            local = Path(root or ".") / self.settings.id
            if not local.is_dir() and not self.settings.revision:
                raise ValueError("Pin model.revision when loading a Hub base")
            self.base_path = resolve_export(
                str(local.resolve()) if local.is_dir() else self.settings.id,
                revision=self.settings.revision,
            )
            self.sdk = Qwen3TTSModel.from_pretrained(
                str(self.base_path),
                device_map=str(self.device),
                dtype=torch.float32,
                attn_implementation="eager",
            )
            self.model = self.sdk.model
            self._configure_qwen()

    def _configure_qwen(self):
        if (
            self.model.tts_model_type != "base"
            or self.model.tokenizer_type != "qwen3_tts_tokenizer_12hz"
        ):
            raise ValueError("Only Qwen3-TTS 12Hz Base single-speaker fine-tuning is supported")
        if self.model.talker.config.num_code_groups != 16:
            raise ValueError("Expected the native 16-codebook Qwen3-TTS 12Hz format")
        self.model.speaker_encoder.requires_grad_(False)
        self.model.speaker_encoder.eval()
        # Qwen's codec wrapper is separate from the trainable talker module.
        self.model.speech_tokenizer.model.requires_grad_(False)
        self.model.speech_tokenizer.model.eval()

    def fit_voice(self, waveform):
        if self.settings.id == "tiny":
            self.voice = torch.tensor([float(np.mean(waveform))])
        else:
            self.model.speaker_encoder.eval()
            with torch.no_grad():
                self.voice = (
                    self.model.extract_speaker_embedding(audio=waveform, sr=24000).detach().cpu()
                )

    def text_ids(self, value):
        text(value, "text")
        if self.settings.id == "tiny":
            ids = torch.tensor(list(value.encode("utf-8")), dtype=torch.long)
        else:
            wrapped = f"<|im_start|>assistant\n{value}<|im_end|>\n<|im_start|>assistant\n"
            ids = self.sdk.processor(text=wrapped, return_tensors="pt")["input_ids"][0][:-5]
            if len(ids) < 4:
                raise ValueError("Unexpected native assistant token layout")
        if len(ids) > self.settings.max_text_tokens:
            raise ValueError("Text exceeds token budget; no silent truncation")
        return ids

    def encode(self, row):
        ids = self.text_ids(row["text"])
        if self.settings.id == "tiny":
            # A deterministic toy codec (NOT Qwen's codec), for synthetic mechanics only.
            frames = np.array_split(row["waveform"], 12)
            codes = torch.tensor(
                [min(31, max(0, int((float(np.mean(frame)) + 1) * 16))) for frame in frames]
            )
        else:
            self.model.speech_tokenizer.model.eval()
            with torch.no_grad():
                codes = (
                    self.model.speech_tokenizer.encode(row["waveform"], sr=24000)
                    .audio_codes[0]
                    .cpu()
                )
            if codes.ndim != 2 or codes.shape[1] != 16 or not len(codes):
                raise ValueError("Invalid native audio codec output")
            if len(ids) + len(codes) + 8 > self.model.talker.config.max_position_embeddings:
                raise ValueError("Text/audio exceed native context; split clips deliberately")
        return {
            "id": row["id"],
            "text": row["text"],
            "text_ids": ids,
            "codes": codes,
            "speaker_embedding": self.voice,
        }

    def loss_weight(self, rows):
        # Equal utterance weighting: first/residual codebook losses have different denominators.
        return len(rows)

    def loss(self, rows):
        if self.settings.id != "tiny":
            from .qwen import native_loss

            return torch.stack([native_loss(self.model, [row]) for row in rows]).mean()
        losses = []
        for row in rows:
            labels = row["codes"].to(self.device)
            inputs = torch.cat((labels.new_tensor([32]), labels[:-1])).unsqueeze(0)
            logits = self.model(inputs, row["text_ids"].unsqueeze(0).to(self.device))[0]
            losses.append(torch.nn.functional.cross_entropy(logits, labels))
        return torch.stack(losses).mean()

    def synthesize(self, value, output):
        ids = self.text_ids(value)
        self.model.eval()
        if self.settings.id == "tiny":
            generated = []
            with torch.inference_mode():
                for _ in range(self.settings.max_new_tokens):
                    inputs = torch.tensor([[32, *generated]], device=self.device)
                    token = int(
                        self.model(inputs, ids.unsqueeze(0).to(self.device))[0, -1].argmax()
                    )
                    generated.append(token)
            waveform = np.repeat((np.asarray(generated, dtype=np.float32) - 16) / 16, 200)
            rate = 24000
        else:
            prompt = {
                "ref_code": [None],
                "ref_spk_embedding": [self.voice.to(self.device)],
                "x_vector_only_mode": [True],
                "icl_mode": [False],
            }
            wavs, rate = self.sdk.generate_voice_clone(
                text=value,
                language="Auto",
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                do_sample=False,
                subtalker_dosample=False,
                max_new_tokens=self.settings.max_new_tokens,
            )
            waveform = wavs[0]
        return write_audio(output, waveform, rate)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.voice is None:
            raise ValueError("Fit a training-only voice reference before saving")
        if self.settings.id == "tiny":
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()},
                str(path / "model.safetensors"),
            )
        else:
            self.model.save_pretrained(path, safe_serialization=True)
            self.sdk.processor.save_pretrained(path)
            # Qwen's loader reads this SDK dictionary, not just HF generation defaults.
            write_json(path / "generation_config.json", self.model.generate_config)
            shutil.copytree(
                self.base_path / "speech_tokenizer", path / "speech_tokenizer", dirs_exist_ok=True
            )
        save_file(
            {"speaker_embedding": self.voice.detach().cpu().contiguous()},
            str(path / "voice.safetensors"),
        )
        write_json(
            path / "advai_metadata.json",
            {
                "schema_version": 1,
                "kind": "tts",
                "model": asdict(self.settings),
                "conditioning": "fixed_training_reference_xvector",
            },
        )
        return path

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **options):
        path = resolve_export(path, **options)
        meta = json.loads((path / "advai_metadata.json").read_text())
        if meta.get("kind") != "tts" or meta.get("schema_version") != 1:
            raise ValueError("Not a supported TTS export")
        obj = cls.__new__(cls)
        obj.settings, obj.device = ModelConfig(**meta["model"]), torch.device(device)
        obj.voice = load_file(str(path / "voice.safetensors"))["speaker_embedding"]
        obj.base_path = path
        if obj.settings.id == "tiny":
            obj.model = TinyTTS().to(obj.device)
            obj.model.load_state_dict(load_file(str(path / "model.safetensors")))
        else:
            from qwen_tts import Qwen3TTSModel

            obj.sdk = Qwen3TTSModel.from_pretrained(
                str(path),
                device_map=str(obj.device),
                dtype=torch.float32,
                attn_implementation="eager",
                local_files_only=True,
            )
            obj.model = obj.sdk.model
            obj._configure_qwen()
        return obj
