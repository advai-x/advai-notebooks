"""Task definitions and hard/soft labels for typed decisions."""

import math

SPLITS = {"train", "validation", "calibration", "test"}
QTYPES = {"choice": 0, "score": 1, "noul": 2}


def validate_tasks(tasks: dict) -> dict:
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Tasks must be a nonempty task_id -> questions mapping")
    for task_id, questions in tasks.items():
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(questions, dict)
            or not questions
        ):
            raise ValueError("Each task needs a string ID and nonempty questions")
        for qid, q in questions.items():
            if not isinstance(qid, str) or not qid or not isinstance(q, dict):
                raise ValueError(f"Invalid question in {task_id}")
            if set(q) - {"type", "instructions", "criteria", "labels"}:
                raise ValueError(f"Unknown question keys in {task_id}/{qid}")
            if (
                q.get("type") not in QTYPES
                or not isinstance(q.get("instructions"), str)
                or not q["instructions"].strip()
            ):
                raise ValueError(f"{task_id}/{qid}: require type and instructions")
            crit = q.get("criteria")
            if q["type"] == "choice":
                if (
                    not isinstance(crit, dict)
                    or len(crit) < 2
                    or any(not isinstance(k, str) or not k.strip() for k in crit)
                    or any(not isinstance(v, str) for v in crit.values())
                ):
                    raise ValueError(f"{qid}: choice needs at least two named criteria")
            elif q["type"] == "score":
                if (
                    not isinstance(crit, list)
                    or len(crit) < 2
                    or any(not isinstance(v, str) or not v.strip() for v in crit)
                ):
                    raise ValueError(f"{qid}: score needs at least two ordered levels")
            elif crit is not None and (
                not isinstance(crit, dict)
                or set(crit) - {"false", "true"}
                or any(not isinstance(v, str) for v in crit.values())
            ):
                raise ValueError(f"{qid}: noul criteria may only contain false and true")
            if "labels" in q:
                labels = q["labels"]
                if (
                    q["type"] != "noul"
                    or not isinstance(labels, dict)
                    or set(labels) != {"false", "true"}
                    or any(not isinstance(v, str) or not v.strip() for v in labels.values())
                    or len({v.strip() for v in labels.values()}) != 2
                ):
                    raise ValueError(f"{qid}: labels must contain distinct false/true descriptions")
    return tasks


def option_keys(question: dict) -> list[str]:
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def target_distribution(question: dict, gold: dict) -> list[float]:
    keys = option_keys(question)
    kind = question["type"]
    if not isinstance(gold, dict) or len(gold) != 1:
        raise ValueError("Gold must contain exactly one typed label or probabilities")
    if "probabilities" in gold:
        probs = gold["probabilities"]
        if not isinstance(probs, dict) or not probs or set(probs) - set(keys):
            raise ValueError("Gold probabilities contain unknown or missing options")
        values = [probs.get(k, 0.0) for k in keys]
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
            for v in values
        ):
            raise ValueError("Gold probabilities must be finite and nonnegative")
        total = sum(values)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Gold probabilities must have positive total mass")
        return [float(v) / total for v in values]
    if set(gold) != {kind}:
        raise ValueError(f"Expected a {kind} label")
    value = gold[kind]
    if kind == "noul":
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("noul label must be a boolean or probability in [0, 1]")
        return [1.0 - float(value), float(value)]
    if kind == "score":
        if type(value) is not int or not 0 <= value < len(keys):
            raise ValueError(
                "score label must be a zero-based integer level; use probabilities for soft labels"
            )
        index = value
    else:
        if not isinstance(value, str) or value not in keys:
            raise ValueError("Unknown choice label")
        index = keys.index(value)
    return [float(i == index) for i in range(len(keys))]
