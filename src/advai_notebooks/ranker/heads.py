"""Native scoring-layer views and the single raw relevance-logit output rule."""


class RelevanceHead:
    def __init__(self, backbone):
        if backbone.model.num_labels != 1:
            raise ValueError("Ranker requires a sequence classifier with one relevance logit")
        self.backbone = backbone

    @property
    def modules(self):
        # Expose all classifier-side modules, including architecture-specific
        # poolers/dropouts, without registering parameters a second time.
        return {
            name: module
            for name, module in self.backbone.native_model.named_children()
            if module is not self.backbone.encoder
        }

    def __call__(self, outputs):
        return outputs["scores"].reshape(-1)


def build_head(backbone):
    return RelevanceHead(backbone)
