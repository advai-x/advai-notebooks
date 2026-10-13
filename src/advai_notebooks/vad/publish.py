"""Explicit VAD inference publication; raw audio and training state stay local."""

from advai_notebooks.common.native_publication import publish as package

from .model import VADModel


def publish(export_dir, repo_id, **options):
    return package(
        export_dir,
        repo_id,
        "vad",
        VADModel,
        "Offline voice-activity detector. Reload with VADModel.load. "
        "Silero updates the selected decoder only; synthetic smoke is not a benchmark. "
        "Review data permissions and retain Silero's MIT notice for native exports.",
        asset_names=("silero_vad.jit",),
        **options,
    )
