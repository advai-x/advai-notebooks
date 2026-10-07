"""A view of the native HF causal decoder, preserving its own forward implementation."""


class DecoderBackbone:
    def __init__(self, model):
        self.model = model.get_base_model() if hasattr(model, "peft_config") else model

    @property
    def module(self):
        return self.model.get_decoder()
