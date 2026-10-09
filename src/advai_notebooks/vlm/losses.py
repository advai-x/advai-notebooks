"""Assistant-token causal cross entropy through the native LLaVA forward."""


def loss(model, rows):
    return model.model(**model.batch(rows)).loss
