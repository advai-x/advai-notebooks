"""HF token representations and pooling, independent of the decision head."""

from torch import nn


class HFOptionBackbone(nn.Module):
    def __init__(self, encoder, pooling="mean"):
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("Backbone pooling must be mean or cls")
        self.encoder, self.pooling = encoder, pooling
        self.output_size = encoder.config.hidden_size

    def forward(self, input_ids, attention_mask, option_mask, token_type_ids=None):
        inputs = {
            "input_ids": input_ids[option_mask],
            "attention_mask": attention_mask[option_mask],
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids[option_mask]
        hidden = self.encoder(**inputs).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        mask = option_mask.unsqueeze(-1).expand(*option_mask.shape, self.output_size)
        return pooled.new_zeros(mask.shape).masked_scatter(mask, pooled)


def build_backbone(encoder, pooling):
    return HFOptionBackbone(encoder, pooling)
