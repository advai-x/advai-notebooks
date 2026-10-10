"""Explicit inference-only speech-to-text publication; never upload recordings."""

from advai_notebooks.common.native_publication import publish as package

from .model import STTModel


def publish(export_dir, repo_id, **options):
    return package(
        export_dir,
        repo_id,
        "stt",
        STTModel,
        "Whisper STT export. Synthetic smoke metrics do not establish recognition quality. "
        "Reload with advai_notebooks.stt.model.STTModel.load. Review base/data licenses independently.",
        pipeline_tag="automatic-speech-recognition",
        **options,
    )
