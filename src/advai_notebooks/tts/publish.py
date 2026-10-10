"""Explicit inference-only TTS publication, including codec/voice weights, not raw audio."""

from advai_notebooks.common.native_publication import publish as package

from .model import TTSModel


def publish(export_dir, repo_id, **options):
    return package(
        export_dir,
        repo_id,
        "tts",
        TTSModel,
        "Single-speaker TTS export. Contains a derived speaker embedding: obtain explicit "
        "speaker authorization before publishing it. No raw training audio is uploaded. "
        "Reload with advai_notebooks.tts.model.TTSModel.load. Tiny is a toy PCM codec, not Qwen speech quality.",
        asset_names=("voice.safetensors",),
        component_dirs=("speech_tokenizer",),
        pipeline_tag="text-to-speech",
        **options,
    )
