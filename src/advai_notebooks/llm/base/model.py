"""Public base model: composition, loading, saving and inference."""

from dataclasses import replace

import torch

from advai_notebooks.llm import native
from advai_notebooks.llm.configuration import GenerationSettings

from .backbones import DecoderBackbone
from .config import ModelConfig
from .heads import LanguageModelHead


class BaseModel:
    def __init__(self, settings=None, device="cpu", root=None, training=None, generation=None):
        self.settings = settings or ModelConfig()
        self.device = torch.device(device)
        self.generation = generation or GenerationSettings()
        self.model, self.tokenizer, self.source, self.base_source, self._temporary = (
            native.initialize(self.settings, training, root, self.device, supports_moe=True)
        )
        self._components()

    def _components(self):
        self.backbone = DecoderBackbone(self.model)
        self.head = LanguageModelHead(self.model)

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **hub_options):
        obj = cls.__new__(cls)
        obj.device = torch.device(device)
        obj.model, obj.tokenizer, meta, obj.base_source = native.load_native(
            path,
            "llm/base",
            obj.device,
            trainable=trainable,
            **hub_options,
        )
        obj.settings = ModelConfig(**meta["model"])
        obj.generation = GenerationSettings(**meta["generation"])
        obj.source, obj._temporary = meta["source"], None
        obj._components()
        return obj

    def save(self, path, *, merged=False):
        return native.save_native(self, path, "llm/base", merged=merged)

    def prepare(self, row):
        from .data import encode_row

        return encode_row(row, self, reject_truncation=True)

    def configure(self, settings, generation):
        """Set experiment context/generation limits without replacing model weights."""
        if settings.chat_template is not None:
            self.tokenizer.chat_template = settings.chat_template
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and settings.max_len > limit:
            raise ValueError("max_len exceeds model context")
        self.settings = replace(
            self.settings, max_len=settings.max_len, chat_template=settings.chat_template
        )
        self.generation = generation

    def generate(self, texts, **options):
        """Return generated continuations, excluding each input prompt."""
        return native.generate(self, texts, **options)
