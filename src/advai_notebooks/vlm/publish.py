"""Explicit inference-only VLM packages, including offline preview."""

from advai_notebooks.common.native_publication import publish as upload

from .model import VLMModel


def publish(export_dir, repo_id, **options):
    return upload(
        export_dir,
        repo_id,
        "vlm",
        VLMModel,
        "Native LLaVA image-question answering; answer-only full/LoRA training. Tiny smoke weights are random, not a usable assistant.\n\nReload with `advai_notebooks.vlm.model.VLMModel.load(repo_id)`. "
        "No raw training data or optimizer state is included. Model licenses "
        "and dataset permissions remain the publisher's responsibility.",
        **options,
    )
