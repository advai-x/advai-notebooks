"""Native ragged option collation and shared single-device runtime."""

from advai_notebooks.common.runtime import amp_context, setup  # noqa: F401


def make_batch(items, pad_id, device):
    import torch

    if not items:
        raise ValueError("Decision batch is empty")
    labeled = ["target" in item for item in items]
    if any(labeled) and not all(labeled):
        raise ValueError("Cannot mix labeled and unlabeled decision items")
    size, options = len(items), max(len(item["options"]) for item in items)
    length = max(len(option["input_ids"]) for item in items for option in item["options"])
    ids = torch.full((size, options, length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    option_mask = torch.zeros((size, options), dtype=torch.bool)
    target = torch.zeros((size, options), dtype=torch.float32) if all(labeled) else None
    has_types = "token_type_ids" in items[0]["options"][0]
    token_types = torch.zeros_like(ids) if has_types else None
    for row, item in enumerate(items):
        if target is not None:
            target[row, : len(item["target"])] = torch.tensor(item["target"])
        for col, option in enumerate(item["options"]):
            n = len(option["input_ids"])
            ids[row, col, :n] = torch.tensor(option["input_ids"])
            attention[row, col, :n] = torch.tensor(option["attention_mask"])
            option_mask[row, col] = True
            if has_types:
                token_types[row, col, :n] = torch.tensor(option["token_type_ids"])
    batch = {
        "input_ids": ids,
        "attention_mask": attention,
        "option_mask": option_mask,
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }
    if target is not None:
        batch["target"] = target
    if has_types:
        batch["token_type_ids"] = token_types
    return {key: value.to(device) for key, value in batch.items()}


def forward(model, batch):
    return model(**{key: value for key, value in batch.items() if key != "target"})
