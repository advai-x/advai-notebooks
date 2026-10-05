"""Ranker-specific pair shuffling and batching."""

import random


def pair_batches(rows, batch_size, seed):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
