"""Task sampling delegates deterministic epoch shuffling to Trainer's sampler."""


def training_rows(prepared):
    rows = prepared["train"]
    if not rows:
        raise ValueError("Training split is empty")
    return rows
