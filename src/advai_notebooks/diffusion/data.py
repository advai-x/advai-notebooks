"""Image/caption records, decoded-image leakage checks and deterministic preprocessing."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from advai_notebooks.common.artifacts import digest


def load_data(cfg):
    source = cfg.path(cfg.data.samples)
    rows = []
    ids, groups, images = set(), {}, {}
    counts = {"train": 0, "validation": 0, "test": 0}
    provenance = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"id", "group_id", "split", "image", "caption"}:
            raise ValueError("Image records require id/group_id/split/image/caption")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError("All diffusion record fields must be nonempty strings")
        if row["id"] in ids or row["split"] not in counts:
            raise ValueError("Duplicate ID or invalid split")
        image_path = Path(row["image"])
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError("Image paths must be relative to the JSONL file")
        image_path = (source.parent / image_path).resolve()
        if not image_path.is_relative_to(source.parent.resolve()):
            raise ValueError("Image path escapes dataset directory")
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            content = digest(
                {"size": rgb.size, "pixels": hashlib.sha256(rgb.tobytes()).hexdigest()}
            )
        split = row["split"]
        if groups.get(row["group_id"], split) != split:
            raise ValueError("Image group crosses splits")
        if images.get(content, split) != split:
            raise ValueError("Duplicate decoded image crosses splits")
        ids.add(row["id"])
        groups[row["group_id"]] = split
        images[content] = split
        counts[split] += 1
        rows.append({**row, "image_path": str(image_path)})
        provenance.append(
            {
                **row,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "decoded_fingerprint": content,
            }
        )
    if any(n == 0 for n in counts.values()):
        raise ValueError("Nonempty train/validation/test splits are required")
    return rows, {
        "rows": len(rows),
        "splits": counts,
        "groups": len(groups),
        "fingerprint": digest(provenance),
    }


def prepare_data(rows, model, cfg):
    model.validate_prompts([r["caption"] for r in rows])
    return {
        split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "test")
    }


def pixels(rows, resolution, device):
    result = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            image = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"),
                (resolution, resolution),
                method=Image.Resampling.BICUBIC,
            )
            values = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
        result.append(torch.from_numpy(values).permute(2, 0, 1))
    return torch.stack(result).to(device)
