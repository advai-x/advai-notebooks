"""Native LLaVA weights, multimodal processor and full/LoRA persistence."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.hub import resolve_export

from .config import ModelConfig, TrainingConfig


class VLMModel:
    def __init__(self, settings=None, device="cpu", root=None, training=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.settings, training = settings or ModelConfig(), training or TrainingConfig()
        self.device = torch.device(device)
        self.method = training.method
        self._base = None
        if self.settings.id == "tiny":
            self.model, self.processor = self._tiny()
        else:
            local = Path(root or ".") / self.settings.id
            model_id = str(local.resolve()) if local.is_dir() else self.settings.id
            if not local.is_dir() and not self.settings.revision:
                raise ValueError("Pin model.revision when loading a Hub base")
            options = {"revision": self.settings.revision, "trust_remote_code": False}
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.float32, **options
            )
            self.processor = AutoProcessor.from_pretrained(model_id, backend="pil", **options)
        if self.model.config.model_type != "llava":
            raise ValueError("This backend supports native LlavaForConditionalGeneration only")
        if not self.processor.chat_template:
            raise ValueError("The native LLaVA processor must have an explicit chat template")
        if self.processor.tokenizer.pad_token_id is None:
            raise ValueError("The native LLaVA tokenizer must define a padding token")
        if self.settings.max_len > self.model.config.text_config.max_position_embeddings:
            raise ValueError("max_len exceeds the native language context")
        if self.settings.freeze_vision:
            self.model.model.vision_tower.requires_grad_(False)
        if self.method == "lora":
            from peft import LoraConfig, get_peft_model

            self._base = TemporaryDirectory(prefix="advai-vlm-base-")
            self.model.save_pretrained(self._base.name, safe_serialization=True)
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=training.lora_rank,
                    lora_alpha=training.lora_alpha,
                    target_modules=r".*language_model.*\.(q_proj|v_proj)$",
                    task_type="CAUSAL_LM",
                    lora_dropout=0.0,
                ),
            )
        self.model.to(device=self.device, dtype=torch.float32)

    def _tiny(self):
        from transformers import (
            CLIPImageProcessorPil,
            CLIPVisionConfig,
            LlamaConfig,
            LlavaConfig,
            LlavaForConditionalGeneration,
            LlavaProcessor,
        )

        from advai_notebooks.common.toy_processor import tokenizer

        tokens = tokenizer()
        vision = CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
        )
        language = LlamaConfig(
            vocab_size=len(tokens),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=self.settings.max_len,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config = LlavaConfig(
            vision_config=vision.to_dict(),
            text_config=language.to_dict(),
            image_token_index=4,
            image_seq_length=4,
            vision_feature_layer=-1,
            pad_token_id=0,
        )
        template = (
            "{% for m in messages %}{{ m['role'] | upper }}: "
            "{% for c in m['content'] %}{% if c['type'] == 'image' %}<image> "
            "{% else %}{{ c['text'] }}{% endif %}{% endfor %}\n{% endfor %}"
            "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
        )
        processor = LlavaProcessor(
            image_processor=CLIPImageProcessorPil(
                size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
            ),
            tokenizer=tokens,
            patch_size=8,
            num_additional_image_tokens=1,
            vision_feature_select_strategy="default",
            chat_template=template,
        )
        return LlavaForConditionalGeneration(config), processor

    def prompt(self, question):
        return self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode(self, row, *, supervise=True):
        from PIL import Image

        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
            prompt = self.prompt(row["prompt"])
            prefix = self.processor(images=image, text=prompt, return_tensors="pt")
            if supervise:
                answer = prompt + row["answer"] + (self.processor.tokenizer.eos_token or "")
                encoded = self.processor(images=image, text=answer, return_tensors="pt")
                boundary = prefix["input_ids"].shape[1]
                if not torch.equal(encoded["input_ids"][:, :boundary], prefix["input_ids"]):
                    raise ValueError("Assistant boundary is unstable for this processor")
                labels = encoded["input_ids"].clone()
                labels[:, :boundary] = -100
                if not (labels[:, 1:] != -100).any():
                    raise ValueError("No supervised assistant tokens")
                encoded["labels"] = labels
            else:
                encoded = prefix
        limit = (
            self.settings.max_len
            if supervise
            else (self.settings.max_len - self.settings.max_new_tokens)
        )
        if encoded["input_ids"].shape[1] > limit:
            raise ValueError("Multimodal input exceeds the configured context budget")
        encoded.pop("token_type_ids", None)  # LLaVA has no segment embeddings.
        return dict(encoded)

    def batch(self, rows):
        pad = self.processor.tokenizer.pad_token_id
        length = max(row["input_ids"].shape[1] for row in rows)
        result = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            if key in {"input_ids", "attention_mask", "labels"}:
                fill = -100 if key == "labels" else pad if key == "input_ids" else 0
                values = [
                    torch.nn.functional.pad(v, (0, length - v.shape[1]), value=fill) for v in values
                ]
            result[key] = torch.cat(values).to(self.device)
        return result

    def loss_weight(self, rows):
        return sum(int((row["labels"][:, 1:] != -100).sum()) for row in rows)

    def chat(self, image, prompt):
        from advai_notebooks.common.media import text

        text(prompt, "prompt")
        encoded = self.encode({"image": str(image), "prompt": prompt}, supervise=False)
        self.model.eval()
        with torch.inference_mode():
            generated = self.model.generate(
                **{k: v.to(self.device) for k, v in encoded.items()},
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        return self.processor.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def save(self, path, *, merged=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model = self.model
        method = self.method
        if merged and method == "lora":
            import copy

            model = copy.deepcopy(self.model).merge_and_unload()
            method = "full"
        if method == "lora":
            model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
        else:
            model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)
        if method == "lora":
            base = self._base.name if self._base is not None else self.base_path
            shutil.copytree(base, path / "base_model", dirs_exist_ok=True)
        write_json(
            path / "advai_metadata.json",
            {
                "schema_version": 1,
                "kind": "vlm",
                "model": asdict(self.settings),
                "method": method,
            },
        )
        return path

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **options):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = resolve_export(path, **options)
        meta = json.loads((path / "advai_metadata.json").read_text())
        if meta.get("kind") != "vlm" or meta.get("schema_version") != 1:
            raise ValueError("Not a supported VLM export")
        obj = cls.__new__(cls)
        obj.settings, obj.method = ModelConfig(**meta["model"]), meta["method"]
        if obj.method not in {"full", "lora"}:
            raise ValueError("Unsupported VLM export method")
        obj.device, obj._base = torch.device(device), None
        obj.base_path = path / "base_model"
        obj.processor = AutoProcessor.from_pretrained(path, local_files_only=True, backend="pil")
        obj.model = AutoModelForImageTextToText.from_pretrained(
            obj.base_path if obj.method == "lora" else path,
            local_files_only=True,
            dtype=torch.float32,
        )
        if obj.model.config.model_type != "llava" or not obj.processor.chat_template:
            raise ValueError("Export must contain native LLaVA weights and a chat template")
        if obj.settings.freeze_vision:
            obj.model.model.vision_tower.requires_grad_(False)
        if obj.method == "lora":
            from peft import PeftModel

            obj.model = PeftModel.from_pretrained(obj.model, path, is_trainable=trainable)
        obj.model.to(device=obj.device, dtype=torch.float32)
        return obj
