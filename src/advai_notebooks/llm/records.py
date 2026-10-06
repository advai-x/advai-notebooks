"""Neutral JSONL reading and grouped split integrity, without supervision semantics."""

import json
import re
from pathlib import Path

from advai_notebooks.common.artifacts import digest


def read_records(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at line {number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Line {number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def check_splits(rows, group_key, content):
    ids, groups, texts = set(), {}, {}
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        for key in ("id", group_key):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        split = row.get("split")
        if split not in counts:
            raise ValueError("split must be train, validation or test")
        if row["id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(row["id"])
        group = row[group_key]
        if group in groups and groups[group] != split:
            raise ValueError(f"{group_key} crosses data splits")
        groups[group] = split
        text = re.sub(r"\s+", " ", content(row)).strip()
        key = digest(text)
        if key in texts and texts[key] != split:
            raise ValueError("Duplicate content crosses data splits")
        texts[key] = split
        counts[split] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("Dataset requires nonempty train, validation and test splits")
    return {"rows": len(rows), "splits": counts, "groups": len(groups), "fingerprint": digest(rows)}
