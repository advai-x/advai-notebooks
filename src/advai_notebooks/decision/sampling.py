"""Decision-specific epoch ordering and weighted accumulation groups."""

import random


def decision_groups(items, settings, epoch):
    order = list(items)
    random.Random(settings.seed + epoch).shuffle(order)
    group_size = settings.batch_size * settings.gradient_accumulation
    return [order[start : start + group_size] for start in range(0, len(order), group_size)]
