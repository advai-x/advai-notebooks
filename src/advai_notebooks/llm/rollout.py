"""Token sampling/log-probability mechanics; no reward or RL objective lives here."""

from dataclasses import dataclass

import torch


@dataclass
class Rollout:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    prompt_width: int
    old_log_probs: torch.Tensor
    completions: list[str]

    def select(self, indices):
        return Rollout(
            self.input_ids[indices],
            self.attention_mask[indices],
            self.completion_mask[indices],
            self.prompt_width,
            self.old_log_probs[indices],
            [self.completions[i] for i in indices.tolist()],
        )


def forward(model, input_ids, attention_mask, *, hidden=False):
    # HF generate uses these positions too. Left padding must not shift real positions.
    positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions,
        use_cache=False,
        output_hidden_states=hidden,
    )


def token_statistics(model, rollout, temperature, value_head=None):
    result = forward(
        model, rollout.input_ids, rollout.attention_mask, hidden=value_head is not None
    )
    start = rollout.prompt_width - 1
    width = rollout.completion_mask.shape[1]
    logits = result.logits[:, start : start + width].float() / temperature
    log_distribution = logits.log_softmax(-1)
    targets = rollout.input_ids[:, rollout.prompt_width :]
    log_probs = log_distribution.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_distribution.exp() * log_distribution).sum(-1)
    values = None
    if value_head is not None:
        states = result.hidden_states[-1][:, start : start + width]
        values = value_head(states).squeeze(-1).float()
    return log_probs, entropy, values


def sample(facade, prompts, *, max_new_tokens, temperature=1.0, greedy=False):
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Sampling requires nonempty prompts")
    old_padding = facade.tokenizer.padding_side
    facade.tokenizer.padding_side = "left"
    try:
        inputs = facade.tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
    finally:
        facade.tokenizer.padding_side = old_padding
    ids = inputs["input_ids"].to(facade.device)
    attention = inputs["attention_mask"].to(facade.device)
    prefix = ids.shape[1]
    if prefix + max_new_tokens > facade.settings.max_len:
        raise ValueError("Prompt plus rollout budget exceeds max_len; no truncation")
    eos = facade.model.generation_config.eos_token_id
    if eos is None:
        eos = facade.tokenizer.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    active = torch.ones(len(prompts), dtype=torch.bool, device=facade.device)
    masks, log_probs = [], []
    previous_mode = facade.model.training
    facade.model.eval()  # Disable dropout for a consistent behavior-policy ratio.
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = forward(facade.model, ids, attention).logits[:, -1].float()
                distribution = (logits / temperature).log_softmax(-1)
                token = (
                    distribution.argmax(-1)
                    if greedy
                    else torch.multinomial(distribution.exp(), 1).squeeze(-1)
                )
                token = torch.where(active, token, facade.tokenizer.pad_token_id)
                masks.append(active.clone())
                log_probs.append(distribution.gather(-1, token[:, None]).squeeze(-1))
                ids = torch.cat([ids, token[:, None]], -1)
                attention = torch.cat([attention, active.long()[:, None]], -1)
                terminal = torch.zeros_like(active)
                for eos_id in eos:
                    terminal |= token == eos_id
                active &= ~terminal
                if not active.any():
                    break
    finally:
        facade.model.train(previous_mode)
    return Rollout(
        ids,
        attention,
        torch.stack(masks, -1),
        prefix,
        torch.stack(log_probs, -1),
        facade.tokenizer.batch_decode(ids[:, prefix:], skip_special_tokens=True),
    )
