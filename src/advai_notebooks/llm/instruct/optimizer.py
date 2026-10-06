"""Task-owned optimizer construction; frozen LoRA base parameters are excluded."""

from advai_notebooks.common.optimization import build_optimizer


def optimizer(model, settings):
    parameters = [p for p in model.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return build_optimizer(parameters, settings)
