"""Load, validate and tokenize decision data without mixing source groups across splits."""

import json
from collections import Counter
from pathlib import Path

from advai_notebooks.common.artifacts import digest

from .contracts import DecisionOption, DecisionRequest  # noqa: F401 (public data API)
from .schema import QTYPES, SPLITS, target_distribution, validate_tasks


def load_data(path: str | Path, task_path: str | Path) -> tuple[list[dict], dict]:
    tasks = load_tasks(task_path)
    records, seen, groups, states = [], set(), {}, {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            required = {"id", "group_id", "split", "task", "state", "gold"}
            if not isinstance(row, dict) or not required <= set(row) or set(row) - required:
                raise ValueError(f"Expected exactly {sorted(required)}")
            for key in ("id", "group_id", "task"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"{key} must be a nonempty string")
            if row["id"] in seen:
                raise ValueError(f"Duplicate record ID {row['id']}")
            if not isinstance(row["split"], str) or row["split"] not in SPLITS:
                raise ValueError("split must be train, validation, calibration or test")
            if (
                not isinstance(row["state"], (str, dict, list))
                or not row["state"]
                or (isinstance(row["state"], str) and not row["state"].strip())
            ):
                raise ValueError("state must be nonempty text, an object or a list")
            if row["task"] not in tasks:
                raise ValueError(f"Unknown task {row['task']}")
            if (
                not isinstance(row["gold"], dict)
                or not row["gold"]
                or set(row["gold"]) - set(tasks[row["task"]])
            ):
                raise ValueError("gold needs labels for at least one known question")
            for qid, gold in row["gold"].items():
                target_distribution(tasks[row["task"]][qid], gold)
            state_key = digest(row["state"])
            for table, key in ((groups, row["group_id"]), (states, state_key)):
                if key in table and table[key] != row["split"]:
                    raise ValueError("Source group or identical state appears in multiple splits")
                table[key] = row["split"]
            seen.add(row["id"])
            records.append(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError("Dataset is empty")
    return records, tasks


def summary(records: list[dict], tasks: dict) -> dict:
    return {
        "records": len(records),
        "source_groups": len({r["group_id"] for r in records}),
        "records_by_split": dict(Counter(r["split"] for r in records)),
        "decisions_by_split": dict(Counter(r["split"] for r in records for _ in r["gold"])),
        "types": dict(Counter(tasks[r["task"]][qid]["type"] for r in records for qid in r["gold"])),
        "fingerprint": digest({"records": records, "tasks": tasks}),
    }


def load_tasks(path):
    return validate_tasks(json.loads(Path(path).read_text(encoding="utf-8")))


def prepare_requests(requests):
    if isinstance(requests, (str, bytes, dict)):
        raise ValueError("At least one DecisionRequest is required")
    try:
        requests = list(requests)
    except TypeError as exc:
        raise ValueError("At least one DecisionRequest is required") from exc
    if not requests or any(not isinstance(request, DecisionRequest) for request in requests):
        raise ValueError("At least one DecisionRequest is required")
    return requests


def requests_from_tasks(state, tasks, task):
    validate_tasks(tasks)
    if task not in tasks:
        raise ValueError(f"Unknown task {task}")
    return [DecisionRequest.from_question(state, question) for question in tasks[task].values()]


def prepare_training_items(records, tasks, model, reject_truncation=True):
    """Training labels are owned here; token features come from the model facade."""
    return _prepare_items(
        records, tasks, lambda request: model.features([request], reject_truncation)[0]
    )


def prepare_items(
    records, tasks, tokenizer, max_len, head_max_len, reject_truncation=True, encoding="pair"
):
    """Compatibility API for callers supplying a tokenizer directly."""
    from .encoding import build_input_encoder

    encoder = build_input_encoder(tokenizer, max_len, head_max_len, reject_truncation, encoding)

    def encode(request):
        result = encoder(request)
        return {"options": result.options, "qtype": request.qtype, "truncated": result.truncated}

    return _prepare_items(records, tasks, encode)


def _prepare_items(records, tasks, encode):
    items = []
    for row in records:
        for qid, gold in row["gold"].items():
            question = tasks[row["task"]][qid]
            target = target_distribution(question, gold)
            try:
                features = encode(DecisionRequest.from_question(row["state"], question))
            except ValueError as exc:
                raise ValueError(f"{row['id']}/{qid}: {exc}") from exc
            items.append(
                features
                | {
                    "qtype": QTYPES[question["type"]],
                    "target": target,
                    "label": max(range(len(target)), key=target.__getitem__),
                    "id": row["id"],
                    "qid": qid,
                    "task": row["task"],
                    "split": row["split"],
                }
            )
    return items
