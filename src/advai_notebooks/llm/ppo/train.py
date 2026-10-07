"""ppo-owned online rollout/update loop; validation is orchestrated by pipeline.py."""

import torch

from advai_notebooks.llm.rollout import token_statistics

from .losses import advantages_and_returns, objective
from .optimizer import parameters
from .reward import score
from .sampling import collect, prompt_batches


def train_epoch(model, reference, rows, cfg, optimizer):
    model.model.eval()  # Gradients still flow; dropout must not change policy ratios.
    reference.model.eval()
    updates, rollout_rewards, statistics = 0, [], []
    rollout_tokens = 0
    for prompts in prompt_batches(rows, cfg.training.batch_size):
        expanded, rollout = collect(model, prompts, cfg.algorithm)
        rewards = torch.tensor(
            [
                score(row, text, cfg.reward)[0]
                for row, text in zip(expanded, rollout.completions, strict=True)
            ],
            device=model.device,
            dtype=torch.float32,
        )
        rollout_rewards.extend(rewards.tolist())
        rollout_tokens += int(rollout.completion_mask.sum())
        with torch.no_grad():
            reference_log_probs, _, _ = token_statistics(
                reference.model, rollout, cfg.algorithm.temperature
            )
            _, _, old_values = token_statistics(
                model.model, rollout, cfg.algorithm.temperature, model.value_head
            )
            shaped = -cfg.algorithm.kl_coefficient * (rollout.old_log_probs - reference_log_probs)
            shaped = shaped * rollout.completion_mask
            terminal = rollout.completion_mask.sum(-1) - 1
            shaped[torch.arange(len(expanded), device=model.device), terminal] += rewards
            advantages, returns = advantages_and_returns(
                shaped,
                old_values,
                rollout.completion_mask,
                cfg.algorithm.gamma,
                cfg.algorithm.gae_lambda,
                cfg.algorithm.normalize_advantages,
            )
        count = len(expanded)
        for _ in range(cfg.algorithm.update_epochs):
            order = torch.randperm(count, device=model.device)
            minibatches = list(order.split(cfg.algorithm.minibatch_size))
            for start in range(0, len(minibatches), cfg.training.gradient_accumulation):
                window = minibatches[start : start + cfg.training.gradient_accumulation]
                window_count = sum(len(indices) for indices in window)
                optimizer.zero_grad(set_to_none=True)
                for indices in window:
                    selected = rollout.select(indices)
                    log_probs, entropy, values = token_statistics(
                        model.model, selected, cfg.algorithm.temperature, model.value_head
                    )
                    loss, stats = objective(
                        log_probs,
                        selected.old_log_probs,
                        advantages[indices],
                        values,
                        old_values[indices],
                        returns[indices],
                        entropy,
                        selected.completion_mask,
                        cfg.algorithm,
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite ppo objective; checkpoint was not committed")
                    (loss * len(indices) / window_count).backward()
                    statistics.append({"loss": float(loss.detach()), **stats})
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters(model), cfg.training.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                updates += 1
                statistics[-1]["gradient_norm"] = float(gradient_norm)
    return {
        "optimizer_steps": updates,
        "rollouts": len(rollout_rewards),
        "completion_tokens": rollout_tokens,
        "rollout_reward_mean": sum(rollout_rewards) / len(rollout_rewards),
        "updates": statistics,
    }
