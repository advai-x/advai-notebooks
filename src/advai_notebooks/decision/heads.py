"""Baseline independent-option scoring head; future interaction heads register here."""

from torch import nn


class IndependentOptionHead(nn.Module):
    def __init__(self, width, hidden_size=None, dropout=0.1):
        super().__init__()
        self.type_embedding = nn.Embedding(3, width)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_size or width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size or width, 1),
        )

    def forward(self, features, option_mask, qtype):
        types = qtype[:, None].expand_as(option_mask)[option_mask]
        features = features[option_mask] + self.type_embedding(types)
        scores = self.scorer(features).squeeze(-1).float()
        return scores.new_full(option_mask.shape, -1e4).masked_scatter(option_mask, scores)


def build_head(name, width, hidden_size=None, dropout=0.1):
    if name != "independent":
        raise ValueError(f"Unsupported decision head: {name}")
    return IndependentOptionHead(width, hidden_size, dropout)
