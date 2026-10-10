"""Audio/transcript records; processor-owned feature extraction and token labels."""

from advai_notebooks.common.audio import load_audio_records


def load_data(cfg):
    return load_audio_records(cfg, 16000)


def prepare_data(rows, model, cfg, *, fit=False):
    del cfg, fit
    prepared = {split: [] for split in ("train", "validation", "test")}
    for row in rows:
        prepared[row["split"]].append(model.encode(row))
    return prepared
