"""Small local-media and numeric-record primitives, without task semantics."""

import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from .artifacts import digest


def records(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Dataset must contain JSONL objects")
    seen, groups = set(), {}
    for row in rows:
        for field in ("id", "group_id", "split"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field}")
        if row["id"] in seen:
            raise ValueError("Duplicate ID")
        seen.add(row["id"])
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Source group crosses splits")
    counts = {
        split: sum(r["split"] == split for r in rows) for split in ("train", "validation", "test")
    }
    if not all(counts.values()):
        raise ValueError("Nonempty train/validation/test splits are required")
    return rows, {"rows": len(rows), "splits": counts, "fingerprint": digest(rows)}


def local_image(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Image reference must be nonempty text")
    root = Path(root).resolve()
    image = Path(value)
    if image.is_absolute() or ".." in image.parts:
        raise ValueError("Image references must be relative to the dataset")
    image = (root / image).resolve()
    if not image.is_relative_to(root) or not image.is_file():
        raise ValueError("Missing or escaping image reference")
    with Image.open(image) as source:
        pixels = source.convert("RGB")
        pixels.load()
    checksum = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
    return image, checksum


def vector(value, size, name):
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in value
        )
    ):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def split_rows(rows):
    return {
        split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "test")
    }
