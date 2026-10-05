"""Embedding sampler and objective plugged into the generic optimizer loop."""

from advai_notebooks.common.retrieval_training import train

from .evaluate import evaluate
from .losses import build_loss
from .metrics import is_improvement, select_metric
from .optimizer import create_optimizer
from .sampling import embedding_batches


def train_model(model, data, cfg, rows, output, signature, resume_state=None):
    return train(
        model,
        cfg,
        lambda epoch: embedding_batches(
            rows, data, cfg.training.batch_size, cfg.training.seed + epoch
        ),
        build_loss(model, cfg.training, data),
        lambda: evaluate(model, data, cfg, "validation")[0],
        output,
        optimizer_factory=create_optimizer,
        metric_selector=select_metric,
        metric_comparator=is_improvement,
        signature=signature,
        resume_state=resume_state,
    )
