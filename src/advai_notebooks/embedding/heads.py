"""Pooling and output transforms, sharing the model's existing module objects."""


def default_head_modules(width):
    from sentence_transformers.sentence_transformer.modules import Normalize, Pooling

    return [Pooling(width, pooling_mode="mean"), Normalize()]


class EmbeddingHead:
    def __init__(self, model, pooling):
        from sentence_transformers.sentence_transformer.modules import Normalize, Pooling

        self.model = model
        pools = [index for index, module in enumerate(model) if isinstance(module, Pooling)]
        if len(pools) != 1:
            raise ValueError("Embedding models must contain exactly one pooling module")
        if pooling != "model":
            model[pools[0]] = Pooling(model[0].get_embedding_dimension(), pooling_mode=pooling)
        model[pools[0]].include_prompt = True
        if not isinstance(model[-1], Normalize):
            model.add_module(str(len(model)), Normalize())

    @property
    def modules(self):
        # These objects remain registered only in SentenceTransformer, retaining
        # original parameter paths and any intermediate Dense/output modules.
        return tuple(self.model)[1:]

    def __call__(self, features):
        for module in self.modules:
            features = module(features)
        return features


def build_head(model, settings):
    return EmbeddingHead(model, settings.pooling)
