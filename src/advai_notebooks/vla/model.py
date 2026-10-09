"""CLIP visual/language features fused with state for continuous action chunks."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.hub import resolve_export
from advai_notebooks.common.media import text, vector

from .config import ModelConfig


class ActionPolicy(torch.nn.Module):
    def __init__(self, encoder, settings):
        super().__init__()
        self.encoder = encoder
        width = encoder.config.projection_dim * 2 + settings.state_dim
        self.head = torch.nn.Sequential(
            torch.nn.Linear(width, settings.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(settings.hidden_size, settings.horizon * settings.action_dim),
        )
        self.settings = settings

    def forward(self, inputs, state):
        vision = self.encoder.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        language = self.encoder.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).pooler_output
        features = torch.cat(
            [self.encoder.visual_projection(vision), self.encoder.text_projection(language), state],
            dim=-1,
        )
        return self.head(features).reshape(-1, self.settings.horizon, self.settings.action_dim)


class VLAModel:
    def __init__(self, settings=None, device="cpu", root=None, training=None):
        from transformers import CLIPModel, CLIPProcessor

        del training
        self.settings = settings or ModelConfig()
        self.device = torch.device(device)
        if self.settings.id == "tiny":
            encoder, self.processor = self._tiny()
        else:
            local = Path(root or ".") / self.settings.id
            model_id = str(local.resolve()) if local.is_dir() else self.settings.id
            if not local.is_dir() and not self.settings.revision:
                raise ValueError("Pin model.revision when loading a Hub base")
            encoder = CLIPModel.from_pretrained(
                model_id, revision=self.settings.revision, dtype=torch.float32
            )
            self.processor = CLIPProcessor.from_pretrained(
                model_id, revision=self.settings.revision, backend="pil"
            )
        if self.settings.max_len > encoder.config.text_config.max_position_embeddings:
            raise ValueError("max_len exceeds the CLIP text context")
        if self.settings.freeze_backbone:
            encoder.requires_grad_(False)
        self.model = ActionPolicy(encoder, self.settings).to(
            device=self.device, dtype=torch.float32
        )
        self.statistics, self.action_spec = None, None

    def _tiny(self):
        from tokenizers.processors import TemplateProcessing
        from transformers import (
            CLIPConfig,
            CLIPImageProcessorPil,
            CLIPModel,
            CLIPProcessor,
            CLIPTextConfig,
            CLIPVisionConfig,
        )

        from advai_notebooks.common.toy_processor import tokenizer

        # CLIP's legacy eos=2 branch pools the largest token ID; keep EOT last.
        tokens = tokenizer(eos_last=True)
        tokens._tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[("<bos>", tokens.bos_token_id), ("<eos>", tokens.eos_token_id)],
        )
        language = CLIPTextConfig(
            vocab_size=len(tokens),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=self.settings.max_len,
            pad_token_id=tokens.pad_token_id,
            bos_token_id=tokens.bos_token_id,
            eos_token_id=tokens.eos_token_id,
        )
        vision = CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
        )
        config = CLIPConfig(
            text_config=language.to_dict(), vision_config=vision.to_dict(), projection_dim=16
        )
        processor = CLIPProcessor(
            image_processor=CLIPImageProcessorPil(
                size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
            ),
            tokenizer=tokens,
        )
        return CLIPModel(config), processor

    def inputs(self, rows):
        from PIL import Image

        images = []
        for row in rows:
            with Image.open(row["image"]) as source:
                images.append(source.convert("RGB"))
        inputs = self.processor(
            images=images,
            text=[r["instruction"] for r in rows],
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        if inputs["input_ids"].shape[1] > self.settings.max_len:
            raise ValueError("Instruction exceeds max_len")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def normalized_state(self, rows):
        states = torch.tensor([r["state"] for r in rows], dtype=torch.float32, device=self.device)
        mean = torch.tensor(self.statistics["state_mean"], device=self.device)
        scale = torch.tensor(self.statistics["state_std"], device=self.device)
        return (states - mean) / scale

    def raw_actions(self, rows):
        prediction = self.model(self.inputs(rows), self.normalized_state(rows))
        scale = torch.tensor(self.statistics["action_std"], device=self.device)
        mean = torch.tensor(self.statistics["action_mean"], device=self.device)
        return prediction * scale + mean

    def loss_weight(self, rows):
        return sum(sum(r["mask"]) * self.settings.action_dim for r in rows)

    def predict(self, image, instruction, state):
        if self.statistics is None:
            raise ValueError("Fit training normalization or load a trained export first")
        text(instruction, "instruction")
        vector(state, self.settings.state_dim, "state")
        self.model.eval()
        with torch.inference_mode():
            actions = (
                self.raw_actions(
                    [{"image": str(image), "instruction": instruction, "state": state}]
                )[0]
                .cpu()
                .tolist()
            )
        return {"actions": actions, "action_spec": self.action_spec}

    def save(self, path):
        if self.statistics is None or self.action_spec is None:
            raise ValueError("An export must include fitted normalization and action semantics")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.encoder.save_pretrained(path / "encoder", safe_serialization=True)
        self.processor.save_pretrained(path / "encoder")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.model.head.state_dict().items()},
            path / "action_head.safetensors",
        )
        write_json(
            path / "advai_metadata.json",
            {
                "schema_version": 1,
                "kind": "vla",
                "model": asdict(self.settings),
                "statistics": self.statistics,
                "action_spec": self.action_spec,
            },
        )
        return path

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **options):
        from transformers import CLIPModel, CLIPProcessor

        del trainable
        path = resolve_export(path, **options)
        meta = json.loads((path / "advai_metadata.json").read_text())
        if meta.get("kind") != "vla" or meta.get("schema_version") != 1:
            raise ValueError("Not a supported VLA export")
        obj = cls.__new__(cls)
        obj.settings, obj.device = ModelConfig(**meta["model"]), torch.device(device)
        encoder = CLIPModel.from_pretrained(
            path / "encoder", local_files_only=True, dtype=torch.float32
        )
        if obj.settings.freeze_backbone:
            encoder.requires_grad_(False)
        obj.processor = CLIPProcessor.from_pretrained(
            path / "encoder", local_files_only=True, backend="pil"
        )
        obj.model = ActionPolicy(encoder, obj.settings)
        obj.model.head.load_state_dict(load_file(path / "action_head.safetensors"))
        obj.model.to(device=obj.device, dtype=torch.float32)
        obj.statistics, obj.action_spec = meta["statistics"], meta["action_spec"]
        return obj
