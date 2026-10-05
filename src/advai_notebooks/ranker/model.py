"""Single-logit CrossEncoder trained with binary relevance labels."""

from pathlib import Path
from tempfile import TemporaryDirectory

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.tiny import write_tiny_checkpoint

from .backbones import build_backbone
from .heads import build_head


class RankerModel:
    def __init__(self, settings, device, root=None):
        import torch
        from sentence_transformers import CrossEncoder

        self.settings, self.device = settings, device
        source = settings.id
        if root is not None and (Path(root) / source).is_dir():
            source = str((Path(root) / source).resolve())
        kwargs = {
            "device": str(device),
            "max_length": settings.max_len,
            "activation_fn": torch.nn.Identity(),
        }
        if source == "tiny":
            with TemporaryDirectory() as folder:
                write_tiny_checkpoint(folder, settings.max_len, classification=True)
                self.model = CrossEncoder(folder, **kwargs)
        else:
            self.model = CrossEncoder(source, revision=settings.revision, **kwargs)
        self._backbone_component = build_backbone(self.model, settings)
        self.head = build_head(self._backbone_component)

    @property
    def backbone(self):
        return self._backbone_component.encoder

    def prepare(self, pairs):
        return [
            (self.settings.query_prefix + q, self.settings.document_prefix + d) for q, d in pairs
        ]

    def features(self, pairs):
        return {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in self.model.preprocess(self.prepare(pairs)).items()
        }

    def logits(self, pairs):
        return self.head(self._backbone_component(self.features(pairs)))

    def check_lengths(self, pairs):
        for start in range(0, len(pairs), 256):
            prepared = self.prepare(pairs[start : start + 256])
            tokens = self.model.tokenizer(
                [p[0] for p in prepared], [p[1] for p in prepared], truncation=False
            )
            if any(len(ids) > self.settings.max_len for ids in tokens["input_ids"]):
                raise ValueError(
                    "Query/document pair exceeds max_len; increase it or disable reject_truncation"
                )

    def score(self, pairs, batch_size=32):
        import torch

        self.model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                values.extend(self.logits(pairs[start : start + batch_size]).float().cpu().tolist())
        if any(not torch.isfinite(torch.tensor(v)) for v in values):
            raise ValueError("Ranker returned non-finite scores")
        return values

    def save(self, path):
        path = Path(path)
        self.model.save_pretrained(str(path), create_model_card=False)
        write_json(
            path / "advai_metadata.json",
            {
                "kind": "ranker",
                "model": vars(self.settings),
                "score_type": "raw_logit",
                "label_rule": "relevance > 0",
            },
        )

    @classmethod
    def load(cls, path, device="cpu"):
        import json

        from advai_notebooks.ranker.config import ModelConfig

        path = Path(path).resolve()
        metadata = json.loads((path / "advai_metadata.json").read_text())["model"]
        metadata.pop("pooling", None)  # Older exports stored the shared embedding setting.
        settings = ModelConfig(**metadata)
        settings.id, settings.revision = str(path), None
        return cls(settings, device)
