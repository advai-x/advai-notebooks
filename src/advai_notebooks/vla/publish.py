"""Explicit inference-only VLA packages, including offline preview."""

from advai_notebooks.common.native_publication import publish as upload

from .model import VLAModel


def publish(export_dir, repo_id, **options):
    return upload(
        export_dir,
        repo_id,
        "vla",
        VLAModel,
        "CLIP/state continuous-action behavior-cloning baseline. Not an OpenVLA checkpoint or a safe robot controller. Validate action units, coordinate frame and operating limits before any deployment.\n\nReload with `advai_notebooks.vla.model.VLAModel.load(repo_id)`. "
        "No raw training data or optimizer state is included. Model licenses "
        "and dataset permissions remain the publisher's responsibility.",
        **options,
    )
