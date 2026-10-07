"""Native vocabulary projection (GRPO has no value head)."""


class LanguageModelHead:
    def __init__(self, model):
        self.model = model

    @property
    def module(self):
        return self.model.get_output_embeddings()
