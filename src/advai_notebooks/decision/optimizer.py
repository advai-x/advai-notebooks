"""Native encoder/head parameter groups and optimizer selection, independent of retrieval trainers."""


def create_optimizer(model, settings):
    import torch

    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder, head = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (encoder if id(parameter) in encoder_ids else head).append(parameter)
    optimizer = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[settings.optimizer]
    return optimizer(
        [{"params": encoder, "lr": settings.encoder_lr}, {"params": head, "lr": settings.head_lr}],
        weight_decay=settings.weight_decay,
    )
