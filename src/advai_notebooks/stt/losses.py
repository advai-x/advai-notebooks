"""Native encoder-decoder token cross-entropy, ignoring padded labels."""


def loss(model, rows):
    return model.model(**model.batch(rows)).loss
