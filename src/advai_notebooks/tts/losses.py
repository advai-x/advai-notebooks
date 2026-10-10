"""Task-owned first-codebook and residual-codebook teacher-forcing objective."""


def loss(model, rows):
    return model.loss(rows)
