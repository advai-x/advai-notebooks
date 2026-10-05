"""Ranker relevance objectives; independent of embedding losses."""


def build_loss(model, settings, data):
    import torch

    if settings.loss != "bce":
        raise ValueError("Unsupported ranker loss")

    def loss_for_batch(batch):
        pairs = [
            (data.queries[r["query_id"]]["text"], data.corpus[r["doc_id"]]["text"]) for r in batch
        ]
        labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=model.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(model.logits(pairs), labels)

    return loss_for_batch
