"""ppo owns trainable parameter selection, including its task-specific heads."""

from advai_notebooks.common.optimization import build_optimizer


def parameters(model):
    result = [p for p in model.model.parameters() if p.requires_grad]
    result += list(model.value_head.parameters())
    return result


def create_optimizer(model, settings):
    return build_optimizer(parameters(model), settings)
