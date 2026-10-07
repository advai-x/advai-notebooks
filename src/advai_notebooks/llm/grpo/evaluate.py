"""Deterministic greedy evaluation; held-out answers never become model inputs."""

from advai_notebooks.llm.rollout import sample

from .metrics import compute
from .reward import score


def evaluate(model, prepared, cfg, split="test"):
    rows = prepared[split]
    predictions = []
    for start in range(0, len(rows), cfg.training.batch_size):
        batch = rows[start : start + cfg.training.batch_size]
        rollout = sample(
            model,
            [r["rendered_prompt"] for r in batch],
            max_new_tokens=cfg.generation.max_new_tokens,
            greedy=True,
        )
        for row, text, length in zip(
            batch, rollout.completions, rollout.completion_mask.sum(-1).tolist(), strict=True
        ):
            value, components = score(row, text, cfg.reward)
            predictions.append(
                {
                    "id": row["id"],
                    "completion": text,
                    "reward": value,
                    "components": components,
                    "completion_tokens": length,
                }
            )
    return {
        "schema_version": 1,
        "kind": cfg.kind,
        "split": split,
        "metrics": compute(predictions),
        "examples": len(predictions),
        "predictions": predictions,
    }
