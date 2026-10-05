"""Optional optimizer constructors; task modules own parameter grouping and selection."""


def build_optimizer(parameters, settings):
    import torch

    optimizer = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[settings.optimizer]
    return optimizer(parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay)
