"""Ranker-owned ranking formulas, records and checkpoint selection."""

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


def _ranking_report(rankings, data, split, cutoffs):
    qids = data.query_ids(split)
    if set(rankings) != set(qids):
        raise ValueError("Rankings must cover exactly the requested split")
    per_query = {}
    for qid in qids:
        docs = [row["doc_id"] for row in rankings[qid]]
        if len(docs) != len(set(docs)) or any(d not in data.corpus for d in docs):
            raise ValueError("Rankings contain duplicate or unknown documents")
        if any(not math.isfinite(row["score"]) for row in rankings[qid]):
            raise ValueError("Ranking scores must be finite")
        labels, positive = data.qrels[qid], data.positives(qid)
        result = {}
        for k in cutoffs:
            top = docs[:k]
            result[f"recall@{k}"] = len(set(top) & positive) / len(positive)
            result[f"mrr@{k}"] = next(
                (1.0 / (i + 1) for i, d in enumerate(top) if d in positive), 0.0
            )
            dcg = sum((2 ** labels.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(top))
            ideal = sum(
                (2**rel - 1) / math.log2(i + 2)
                for i, rel in enumerate(sorted(labels.values(), reverse=True)[:k])
            )
            result[f"ndcg@{k}"] = dcg / ideal
        per_query[qid] = result
    means = {
        key: sum(row[key] for row in per_query.values()) / len(qids)
        for key in next(iter(per_query.values()))
    }
    return {
        "n_queries": len(qids),
        "metrics": means,
        "per_query": per_query,
        "unjudged_policy": "zero_gain_for_offline_metrics",
    }


def metric_direction(name):
    try:
        family, cutoff = name.split("@")
        valid = family in {"recall", "mrr", "ndcg"} and cutoff.isdigit() and int(cutoff) > 0
    except (AttributeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"Unsupported ranking metric: {name}")
    return "maximize"


def metric_records(report):
    """Ranking means average queries; each per-query value has sample count one."""
    records = [
        metric_record(
            name, value, report["n_queries"], {"scope": "overall"}, metric_direction(name)
        )
        for name, value in report["metrics"].items()
    ]
    for qid, values in report["per_query"].items():
        records.extend(
            metric_record(name, value, 1, {"scope": "query", "id": qid}, metric_direction(name))
            for name, value in values.items()
        )
    return records


def evaluate_rankings(rankings, data, split, cutoffs):
    report = _ranking_report(rankings, data, split, cutoffs)
    return report | {
        "task": "ranker",
        "metric_schema_version": 1,
        "metric_records": metric_records(report),
    }
