"""Single-speaker audio/transcript pairs with a fixed training-only voice reference."""

from advai_notebooks.common.audio import load_audio_records, local_audio
from advai_notebooks.common.media import text


def load_data(cfg):
    rows, summary = load_audio_records(cfg, 24000)
    speakers = {text(row.get("speaker_id"), "speaker_id") for row in rows}
    if len(speakers) != 1:
        raise ValueError("This TTS backend supports exactly one speaker")
    _, reference, hashes = local_audio(
        cfg.path(cfg.data.samples).parent,
        cfg.data.reference_audio,
        24000,
        max_duration=cfg.data.max_duration,
        min_duration=cfg.data.min_duration,
    )
    if not any(row["split"] == "train" and row["audio_fingerprint"] == hashes[1] for row in rows):
        raise ValueError("Voice reference must be an audio clip from the training split")
    summary.update(speaker_id=next(iter(speakers)), reference_fingerprint=hashes[1])
    for row in rows:
        row["reference_waveform"] = reference
    return rows, summary


def prepare_data(rows, model, cfg, *, fit=False):
    del cfg
    # The frozen reference encoder and codec remain in eval mode even during training.
    if fit:
        model.fit_voice(rows[0]["reference_waveform"])
    prepared = {split: [] for split in ("train", "validation", "test")}
    for row in rows:
        prepared[row["split"]].append(model.encode(row))
    return prepared
