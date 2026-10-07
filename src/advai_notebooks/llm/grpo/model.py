"""grpo model facade: native actor assembly, load/save and generate/chat inference."""

import copy

from advai_notebooks.llm import native
from advai_notebooks.llm.configuration import GenerationSettings, ModelSettings

from .backbones import DecoderBackbone
from .data import validate_messages
from .heads import LanguageModelHead


class GRPOModel:
    def __init__(self, settings, device="cpu", root=None, *, training=None, generation=None):
        self.settings, self.device = settings, device
        self.generation = generation or GenerationSettings()
        (self.model, self.tokenizer, self.source, self.base_source, self._temporary_base) = (
            native.initialize(settings, training, root, device)
        )
        self._components()

    def _components(self):
        self.backbone = DecoderBackbone(self.model)
        self.language_head = LanguageModelHead(self.model)

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **hub_options):
        self = cls.__new__(cls)
        self.model, self.tokenizer, meta, self.base_source = native.load_native(
            path, "llm/grpo", device, trainable=trainable, **hub_options
        )
        self.settings, self.device = ModelSettings(**meta["model"]), device
        self.source, self._temporary_base = meta["source"], None
        self.generation = GenerationSettings(**meta["generation"])
        self._components()

        self.model.eval()
        return self

    def save(self, path, *, merged=False):
        root = native.save_native(self, path, "llm/grpo", merged=merged)

        return root

    def frozen_reference(self):
        reference = copy.copy(self)
        reference.model = copy.deepcopy(self.model).eval()
        reference.model.requires_grad_(False)
        reference.backbone = DecoderBackbone(reference.model)
        reference.language_head = LanguageModelHead(reference.model)
        return reference

    def configure(self, settings, generation):
        maximum = getattr(self.model.config, "max_position_embeddings", None)
        if maximum and settings.max_len > maximum:
            raise ValueError("Configured max_len exceeds model context")
        self.settings = ModelSettings(**{**vars(self.settings), "max_len": settings.max_len})
        self.generation = generation
        if settings.chat_template is not None:
            self.tokenizer.chat_template = settings.chat_template

    def prepare(self, row):
        if "prompt" in row:
            return row["prompt"]
        validate_messages(row["messages"])
        if not self.tokenizer.chat_template:
            raise ValueError("Message inputs require model.chat_template")
        return self.tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompts, *, max_new_tokens=None, **overrides):
        return native.generate(self, prompts, max_new_tokens=max_new_tokens, **overrides)

    def chat(self, messages, *, max_new_tokens=None, **overrides):
        prompt = self.prepare({"messages": messages})
        return self.generate([prompt], max_new_tokens=max_new_tokens, **overrides)[0]
