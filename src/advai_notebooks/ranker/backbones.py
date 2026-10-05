"""Views of the pair encoder and its native sequence-classification forward."""


class PairBackbone:
    def __init__(self, model, max_len):
        self.model = model
        maximum = getattr(self.native_model.config, "max_position_embeddings", None)
        if maximum and max_len > maximum:
            raise ValueError(f"max_len={max_len} exceeds model capacity {maximum}")

    @property
    def native_model(self):
        return self.model.model

    @property
    def encoder(self):
        return self.native_model.base_model

    def __call__(self, features):
        # Native model families own the sequence of encoder, pooler, dropout
        # and classification layers. Keep that path intact across HF architectures.
        return self.model(features)


def build_backbone(model, settings):
    return PairBackbone(model, settings.max_len)
