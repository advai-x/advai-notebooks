"""Instruct-owned token-weighted metrics and validation checkpoint selection."""

import math


def report(total_nll, tokens, examples):
    if tokens < 1 or not math.isfinite(total_nll) or total_nll < 0:
        raise ValueError("Invalid supervised token statistics")
    nll = total_nll / tokens
    perplexity = math.exp(nll) if nll < 709 else None
    return {
        "task": "llm/instruct",
        "metric_schema_version": 1,
        "metrics": {"nll": nll, "perplexity": perplexity},
        "supervised_tokens": tokens,
        "examples": examples,
        "metric_records": [
            {"name": name, "value": value, "n": tokens, "group": "overall", "direction": "lower"}
            for name, value in (("nll", nll), ("perplexity", perplexity))
        ],
    }


def selection_value(result):
    if result.get("task") != "llm/instruct" or result.get("metric_schema_version") != 1:
        raise ValueError("Metric report belongs to another task or schema")
    matches = [
        r
        for r in result.get("metric_records", [])
        if r.get("name") == "nll" and r.get("group") == "overall"
    ]
    if len(matches) != 1:
        raise ValueError("Expected one overall nll record")
    record = matches[0]
    value = record.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or record.get("direction") != "lower"
        or type(record.get("n")) is not int
        or record["n"] < 1
        or record["n"] != result.get("supervised_tokens")
        or result.get("metrics", {}).get("nll") != value
    ):
        raise ValueError("Invalid nll selection record")
    return value


def improves(candidate, selected):
    return selection_value(candidate) < selection_value(selected)


def add_generation_metrics(result, predictions, names):
    import json

    n = len(predictions)
    if n < 1:
        raise ValueError("Generation evaluation has no conversations")
    for name in names:
        if name == "exact_match":
            value = (
                sum(row["prediction"].strip() == row["reference"].strip() for row in predictions)
                / n
            )
        elif name == "json_validity":
            valid = 0
            for row in predictions:
                try:
                    json.loads(row["prediction"])
                    valid += 1
                except json.JSONDecodeError:
                    pass
            value = valid / n
        else:
            raise ValueError("Unsupported generation metric")
        result["metrics"][name] = value
        result["metric_records"].append(
            {"name": name, "value": value, "n": n, "group": "generation", "direction": "higher"}
        )
    return result
