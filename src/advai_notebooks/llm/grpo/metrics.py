"""grpo-local reward, answer accuracy and completion-length evaluation metrics."""


def compute(predictions):
    if not predictions:
        raise ValueError("Evaluation requires nonempty predictions")
    count = len(predictions)
    return {
        "reward_mean": sum(p["reward"] for p in predictions) / count,
        "exact_match": sum(p["components"]["exact_match"] for p in predictions) / count,
        "completion_tokens_mean": sum(p["completion_tokens"] for p in predictions) / count,
    }


def selection_score(report):
    if report.get("kind") != "llm/grpo" or report.get("split") != "validation":
        raise ValueError("Checkpoint selection requires grpo validation metrics")
    return report["metrics"]["reward_mean"]
