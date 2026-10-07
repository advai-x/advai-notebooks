"""grpo prompt/reference records; references are used only by rewards, never policy inputs."""

from advai_notebooks.llm.records import check_splits, read_records


def load_data(cfg):
    rows = read_records(cfg.path(cfg.data.samples))
    for row in rows:
        if set(row) - {"id", "group_id", "split", "prompt", "messages", "reference", "metadata"}:
            raise ValueError("Unknown RL record fields")
        if ("prompt" in row) == ("messages" in row):
            raise ValueError("Provide exactly one of prompt or messages")
        if "prompt" in row:
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ValueError("prompt must be nonempty text")
        else:
            validate_messages(row["messages"])
        if "reference" in row and not isinstance(row["reference"], str):
            raise ValueError("reference must be text")
        if (cfg.reward.exact_match_weight or cfg.reward.token_overlap_weight) and not row.get(
            "reference", ""
        ).strip():
            raise ValueError("This reward requires a nonempty reference")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError("metadata must be an object")
    summary = check_splits(rows, "group_id", lambda row: row.get("prompt") or repr(row["messages"]))
    return rows, summary


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message needs role/content")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Message content must be nonempty text")
    history = messages[1:] if messages[0]["role"] == "system" else messages
    if (
        not history
        or any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(history))
        or history[-1]["role"] != "user"
    ):
        raise ValueError("Messages must alternate user/assistant and end in user")


def prepare_data(rows, model, cfg):
    result = {split: [] for split in ("train", "validation", "test")}
    budget = max(cfg.algorithm.max_new_tokens, cfg.generation.max_new_tokens)
    for row in rows:
        prompt = model.prepare(row)
        ids = model.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not ids or len(ids) + budget > cfg.model.max_len:
            raise ValueError(f"Prompt {row['id']} plus generation budget exceeds max_len")
        result[row["split"]].append({**row, "rendered_prompt": prompt})
    return result
