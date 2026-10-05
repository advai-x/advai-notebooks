"""Embedding encoder view over the original SentenceTransformer module."""


class TransformerBackbone:
    def __init__(self, module, max_len):
        from sentence_transformers.sentence_transformer.modules import Transformer

        if not isinstance(module, Transformer):
            raise ValueError(
                "Only text Transformer encoders with single-vector pooling are supported"
            )
        self.module = module
        maximum = getattr(self.encoder.config, "max_position_embeddings", None)
        if maximum and max_len > maximum:
            raise ValueError(f"max_len={max_len} exceeds model capacity {maximum}")

    @property
    def encoder(self):
        return self.module.auto_model

    def __call__(self, features):
        return self.module(features)


def build_backbone(model, settings):
    return TransformerBackbone(model[0], settings.max_len)
