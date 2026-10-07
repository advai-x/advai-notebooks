"""ppo-owned deterministic/verifiable rewards and a custom scalar callback."""

import importlib
import json
import math
from collections import Counter


def normalize(text):
    return " ".join(text.lower().split())


def score(row, completion, settings):
    target, actual = normalize(row.get("reference", "")), normalize(completion)
    reference_words, words = Counter(target.split()), Counter(actual.split())
    overlap = sum((reference_words & words).values())
    denom = sum(reference_words.values()) + sum(words.values())
    try:
        json.loads(completion)
        json_valid = 1.0
    except (ValueError, TypeError):
        json_valid = 0.0
    components = {
        "exact_match": float(actual == target and bool(target)),
        "token_overlap": 2 * overlap / denom if denom else 0.0,
        "json_validity": json_valid,
        "length": max(
            0.0, 1 - abs(len(actual.split()) - settings.target_length) / settings.target_length
        ),
    }
    value = sum(getattr(settings, name + "_weight") * reward for name, reward in components.items())
    if settings.custom_function:
        module, name = settings.custom_function.split(":")
        custom = getattr(importlib.import_module(module), name)(row, completion)
        if isinstance(custom, bool) or not isinstance(custom, (int, float)):
            raise ValueError("Custom reward must return a finite scalar number")
        components["custom"] = float(custom)
        value += custom
    if not math.isfinite(value):
        raise ValueError("Reward must be finite")
    return float(value), components
