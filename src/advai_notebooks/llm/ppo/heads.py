"""Native vocabulary projection and a trainable state-value head."""


class LanguageModelHead:
    def __init__(self, model):
        self.model = model

    @property
    def module(self):
        return self.model.get_output_embeddings()


def build_value_head(model, device):
    import torch

    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head
