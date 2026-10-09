"""Explicit inference-only VM packages, including offline preview."""

from advai_notebooks.common.native_publication import publish as upload

from .model import VMModel


def publish(export_dir, repo_id, **options):
    return upload(
        export_dir,
        repo_id,
        "vm",
        VMModel,
        "Action-conditioned residual state-space MLP. Numerical observations only, not a video or image world model. Rollout errors must be evaluated on held-out trajectories.\n\nReload with `advai_notebooks.vm.model.VMModel.load(repo_id)`. "
        "No raw training data or optimizer state is included. Model licenses "
        "and dataset permissions remain the publisher's responsibility.",
        **options,
    )
