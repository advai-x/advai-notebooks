"""Diffusion-owned formulas and the project's metric-record contract."""

import math


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
        or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in group.items())
    ):
        raise ValueError("Metric group requires a scope and nonempty string fields")
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
    if report.get("metric_schema_version") != 1:
        raise ValueError("Expected metric_schema_version=1")
    records = report.get("metric_records")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Expected metric records")
    matches = [
        r for r in records if r.get("name") == name and r.get("group") == {"scope": "overall"}
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one overall selection metric")
    return metric_record(**matches[0])


def is_improvement(value, best, direction):
    if direction == "minimize":
        return value < best
    if direction == "maximize":
        return value > best
    raise ValueError("Unsupported metric direction")


def compute(predictions):
    if not predictions:
        raise ValueError("Metrics require nonempty predictions")
    value = sum(r["denoising_mse"] for r in predictions) / len(predictions)
    record = metric_record(
        "denoising_mse", value, len(predictions), {"scope": "overall"}, "minimize"
    )
    return {
        "denoising_mse": value,
        "task": "diffusion",
        "metric_schema_version": 1,
        "metric_records": [record],
    }
