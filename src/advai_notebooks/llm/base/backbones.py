"""A reference view of the original decoder, preserving native forward semantics."""


class DecoderBackbone:
    def __init__(self, model):
        self.model = model.get_base_model() if hasattr(model, "peft_config") else model

    @property
    def module(self):
        return self.model.get_decoder()

    @property
    def config(self):
        return self.model.config
