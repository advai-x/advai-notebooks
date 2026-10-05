"""Decision metrics over unrounded probabilities and fixed option order."""

import math
from collections import defaultdict

DIRECTIONS = {
    "accuracy": "maximize",
    "nll": "minimize",
    "brier": "minimize",
    "ece": "minimize",
    "normalized_score_mae": "minimize",
}


def metric_direction(name):
    if name not in DIRECTIONS:
        raise ValueError(f"Unsupported decision metric: {name}")
    return DIRECTIONS[name]


def _metrics(rows, bins=10):
    if not rows:
        raise ValueError("Cannot evaluate an empty set")
    correct, nll, brier, confidence, score_error = [], [], [], [], []
    for row in rows:
        p, target = row["probabilities"], row["target"]
        if (
            len(p) != len(target)
            or not p
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
            or not math.isclose(sum(p), 1.0, abs_tol=1e-5)
        ):
            raise ValueError("Invalid prediction probabilities")
        pred = max(range(len(p)), key=p.__getitem__)
        gold = max(range(len(target)), key=target.__getitem__)
        correct.append(int(pred == gold))
        confidence.append(max(p))
        nll.append(-sum(t * math.log(max(v, 1e-12)) for t, v in zip(target, p)))
        brier.append(sum((v - t) ** 2 for v, t in zip(p, target)))
        if row["qtype"] == 1:
            predicted = sum(i * v for i, v in enumerate(p))
            expected = sum(i * t for i, t in enumerate(target))
            score_error.append(abs(predicted - expected) / (len(p) - 1))
    ece = 0.0
    for index in range(bins):
        selected = [
            i for i, conf in enumerate(confidence) if min(bins - 1, int(conf * bins)) == index
        ]
        if selected:
            gap = abs(sum(confidence[i] for i in selected) - sum(correct[i] for i in selected))
            ece += gap / len(rows)
    result = {
        "n": len(rows),
        "accuracy": sum(correct) / len(rows),
        "nll": sum(nll) / len(rows),
        "brier": sum(brier) / len(rows),
        "ece": ece,
    }
    if score_error:
        result["normalized_score_mae"] = sum(score_error) / len(score_error)
    return result


def evaluate_predictions(rows):
    grouped, by_type = defaultdict(list), defaultdict(list)
    for row in rows:
        grouped[f"{row['task']}/{row['qid']}"].append(row)
        by_type[{0: "choice", 1: "score", 2: "noul"}[row["qtype"]]].append(row)
    result = {
        "overall": _metrics(rows),
        "by_question": {k: _metrics(v) for k, v in grouped.items()},
        "by_type": {k: _metrics(v) for k, v in by_type.items()},
    }

    records = _records(result["overall"], rows, {"scope": "overall"})
    for qid, values in grouped.items():
        records.extend(
            _records(result["by_question"][qid], values, {"scope": "question", "id": qid})
        )
    for kind, values in by_type.items():
        records.extend(_records(result["by_type"][kind], values, {"scope": "type", "id": kind}))
    return result | {"task": "decision", "metric_schema_version": 1, "metric_records": records}


def _records(metrics, rows, group):
    return [
        metric_record(
            name,
            value,
            sum(row["qtype"] == 1 for row in rows) if name == "normalized_score_mae" else len(rows),
            group,
            metric_direction(name),
        )
        for name, value in metrics.items()
        if name != "n"
    ]


def metric_record(name, value, n, group, direction):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Metric name must be nonempty")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Metric value must be finite")
    if type(n) is not int or n < 1:
        raise ValueError("Metric sample count must be a positive integer")
    if (
        not isinstance(group, dict)
        or not group.get("scope")
        or any(
            not isinstance(key, str) or not isinstance(item, str) or not item
            for key, item in group.items()
        )
    ):
        raise ValueError("Metric group must contain a scope and nonempty string fields")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("Metric direction must be maximize or minimize")
    return {
        "name": name,
        "value": float(value),
        "n": n,
        "group": dict(group),
        "direction": direction,
    }


def select_metric(report, name):
    """Select exactly one overall metric; never infer direction from its name."""
    if report.get("metric_schema_version") != 1:
        raise ValueError("Expected metric_schema_version=1")
    records = report.get("metric_records")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("Expected a list of metric records")
    matches = [
        record
        for record in records
        if record.get("name") == name and record.get("group") == {"scope": "overall"}
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one overall metric: {name}")
    return metric_record(**matches[0])


def is_improvement(value, best, direction):
    if direction == "maximize":
        return value > best
    if direction == "minimize":
        return value < best
    raise ValueError("Metric direction must be maximize or minimize")
