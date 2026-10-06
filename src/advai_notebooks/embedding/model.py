"""Single-vector SentenceTransformer adapter with consistent prefix handling."""

from pathlib import Path
from tempfile import TemporaryDirectory

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.tiny import write_tiny_checkpoint

from .backbones import build_backbone
from .heads import build_head, default_head_modules


class EmbeddingModel:
    def __init__(self, settings, device, root=None):
        import torch
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import Transformer

        self.settings, self.device = settings, device
        source = settings.id
        if root is not None and (Path(root) / source).is_dir():
            source = str((Path(root) / source).resolve())
        if source == "tiny":
            with TemporaryDirectory() as folder:
                write_tiny_checkpoint(folder, settings.max_len)
                transformer = Transformer(folder, max_seq_length=settings.max_len)
                self.model = SentenceTransformer(
                    modules=[
                        transformer,
                        *default_head_modules(transformer.get_embedding_dimension()),
                    ],
                    device=str(device),
                )
        else:
            self.model = SentenceTransformer(source, revision=settings.revision, device=str(device))
        self._backbone_component = build_backbone(self.model, settings)
        self.head = build_head(self.model, settings)
        self.model.max_seq_length = settings.max_len
        self.model.prompts = {"query": settings.query_prefix, "document": settings.document_prefix}
        self.model.default_prompt_name = None
        self.model.similarity_fn_name = "dot"
        self.model.to(torch.device(device))

    @property
    def backbone(self):
        return self._backbone_component.encoder

    def prepare(self, texts, role):
        prefix = self.settings.query_prefix if role == "query" else self.settings.document_prefix
        return [prefix + text for text in texts]

    def features(self, texts, role):
        return {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in self.model.preprocess(self.prepare(texts, role)).items()
        }

    def check_lengths(self, texts, role):
        tokenizer = self.model.tokenizer
        for start in range(0, len(texts), 256):
            tokens = tokenizer(self.prepare(texts[start : start + 256], role), truncation=False)
            if any(len(ids) > self.settings.max_len for ids in tokens["input_ids"]):
                raise ValueError(
                    f"{role} input exceeds max_len; increase it or disable reject_truncation"
                )

    def encode(self, texts, role, batch_size=32):
        import torch

        self.model.eval()
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                vectors.append(
                    self.head(
                        self._backbone_component(
                            self.features(texts[start : start + batch_size], role)
                        )
                    )["sentence_embedding"]
                    .float()
                    .cpu()
                )
        if not vectors:
            return torch.empty((0, self.model.get_sentence_embedding_dimension()))
        result = torch.cat(vectors)
        if not torch.isfinite(result).all():
            raise ValueError("Embedding model returned non-finite vectors")
        return result

    def save(self, path):
        path = Path(path)
        self.model.save_pretrained(str(path), create_model_card=False)
        write_json(
            path / "advai_metadata.json",
            {
                "kind": "embedding",
                "model": vars(self.settings),
                "normalized": True,
                "similarity": "dot",
                "prefixes_applied_once": True,
            },
        )

    @classmethod
    def load(cls, path, device="cpu", *, revision=None, token=None, cache_dir=None,
             local_files_only=False):
        import json

        from advai_notebooks.common.hub import resolve_export
        from advai_notebooks.embedding.config import ModelConfig

        path = resolve_export(path, revision=revision, token=token, cache_dir=cache_dir,
                              local_files_only=local_files_only)
        settings = ModelConfig(**json.loads((path / "advai_metadata.json").read_text())["model"])
        settings.id, settings.revision, settings.pooling = str(path), None, "model"
        return cls(settings, device)
