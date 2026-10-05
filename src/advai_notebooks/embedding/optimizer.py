"""Embedding optimizer entry point; customize parameter groups here."""

from advai_notebooks.common.optimization import build_optimizer


def create_optimizer(model, settings):
    return build_optimizer(model.model.parameters(), settings)
