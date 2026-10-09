"""Single-image question/answer records and answer-only token supervision."""

from advai_notebooks.common.artifacts import digest
from advai_notebooks.common.media import local_image, records, text


def load_data(cfg):
    path = cfg.path(cfg.data.samples)
    rows, summary = records(path)
    seen = {}
    checksums = []
    for row in rows:
        text(row.get("prompt"), "prompt")
        text(row.get("answer"), "answer")
        image, checksum = local_image(path.parent, row.get("image"))
        previous = seen.setdefault(checksum, row["split"])
        if previous != row["split"]:
            raise ValueError("Duplicate decoded image crosses splits")
        checksums.append(checksum)
        row["image"] = str(image)
    summary["media_fingerprint"] = digest(checksums)
    return rows, summary


def prepare_data(rows, model, cfg, *, fit=False):
    del cfg, fit
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        splits[row["split"]].append(model.encode(row))
    return splits
