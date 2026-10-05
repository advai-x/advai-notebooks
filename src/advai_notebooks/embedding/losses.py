"""Embedding loss factory and contrastive batch preparation."""


def build_loss(adapter, settings, data):
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MultipleNegativesRankingLoss,
    )

    kwargs = {"model": adapter.model, "scale": settings.scale}
    criterion = (
        CachedMultipleNegativesRankingLoss(
            **kwargs, mini_batch_size=settings.cached_mini_batch_size
        )
        if settings.loss == "cached_mnrl"
        else MultipleNegativesRankingLoss(**kwargs)
    )

    def loss_for_batch(batch):
        features = [
            adapter.features([data.queries[r["query_id"]]["text"] for r in batch], "query"),
            adapter.features([data.corpus[r["positive_id"]]["text"] for r in batch], "document"),
        ]
        for i in range(len(batch[0]["negative_ids"])):
            features.append(
                adapter.features(
                    [data.corpus[r["negative_ids"][i]]["text"] for r in batch], "document"
                )
            )
        return criterion(features, None)

    return loss_for_batch
