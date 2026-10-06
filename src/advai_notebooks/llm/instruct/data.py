"""Conversation SFT with assistant-only labels and verifiable template boundaries."""

import json
import re

from advai_notebooks.llm.native import collate
from advai_notebooks.llm.records import check_splits, read_records


def validate_messages(messages, training=True):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty conversation")
    expected = "user"
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message requires role and content only")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Message content must be a nonempty string")
        if i == 0 and message["role"] == "system":
            continue
        if message["role"] != expected:
            raise ValueError("Conversation must alternate user and assistant after optional system")
        expected = "assistant" if expected == "user" else "user"
    last = "assistant" if training else "user"
    if messages[-1]["role"] != last:
        raise ValueError(f"Conversation must end with {last}")


def load_data(cfg):
    rows = read_records(cfg.path(cfg.data.samples))
    for row in rows:
        if "messages" not in row and "prompt" in row and "completion" in row:
            row["messages"] = [
                {"role": "user", "content": row.pop("prompt")},
                {"role": "assistant", "content": row.pop("completion")},
            ]
        elif "prompt" in row or "completion" in row:
            raise ValueError("Use messages or prompt/completion, not both")
        validate_messages(row.get("messages"))
    summary = check_splits(
        rows, "conversation_id", lambda row: json.dumps(row["messages"], ensure_ascii=False)
    )
    return rows, summary


def assistant_encoding(messages, tokenizer):
    template = tokenizer.get_chat_template()
    if not template:
        raise ValueError("An explicit chat template is required")
    if not tokenizer.is_fast:
        raise ValueError("Assistant masking requires a fast tokenizer with offsets")
    if re.search(r"{%[-+]?\s*generation\b", template):
        from transformers.utils.chat_template_utils import render_jinja_template

        rendered, intervals = render_jinja_template(
            [messages],
            chat_template=template,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            **tokenizer.special_tokens_map,
        )
        encoded = tokenizer(rendered[0], add_special_tokens=False, return_offsets_mapping=True)
        mask = [0] * len(encoded["input_ids"])
        # char_to_token(end-1) can be None for trailing whitespace. Mapping intervals
        # directly prevents the SDK mask from spilling into subsequent user turns.
        for start, end in intervals[0]:
            for j, (a, b) in enumerate(encoded["offset_mapping"]):
                if b > start and a < end:
                    if a < start or b > end:
                        raise ValueError("Token crosses assistant generation boundary")
                    mask[j] = 1
        return encoded["input_ids"], mask
    # Templates without generation blocks can work if turn prefixes remain stable.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [0] * len(encoded["input_ids"])
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True
        )
        completed = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        if not text.startswith(completed) or not completed.startswith(prefix):
            raise ValueError("Chat template prefixes are unstable; configure generation blocks")
        start, end = len(prefix), len(completed)
        # Never silently supervise a role marker if add_generation_prompt was ignored.
        if not completed[start:].lstrip().startswith(message["content"]):
            raise ValueError("Template lacks an aligned assistant generation prefix")
        for j, (a, b) in enumerate(encoded["offset_mapping"]):
            if b > start and a < end:
                if a < start or b > end:
                    raise ValueError("Token crosses assistant boundary; use generation blocks")
                mask[j] = 1
    return encoded["input_ids"], mask


def encode_row(row, model, reject_truncation=True):
    validate_messages(row["messages"])
    ids, mask = assistant_encoding(row["messages"], model.tokenizer)
    original = len(ids)
    if len(ids) > model.settings.max_len:
        if reject_truncation:
            raise ValueError("Conversation exceeds max_len; truncation would remove supervision")
        ids, mask = ids[: model.settings.max_len], mask[: model.settings.max_len]
    labels = [token if supervised else -100 for token, supervised in zip(ids, mask)]
    if not any(token != -100 for token in labels[1:]):
        raise ValueError("Conversation has no supervised assistant token after encoding/truncation")
    return [
        {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
            "sample_id": row["id"],
            "truncated": original > len(ids),
            "dropped_tokens": max(0, original - len(ids)),
        }
    ]


def prepare_data(rows, model, cfg):
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        splits[row["split"]].extend(encode_row(row, model, cfg.data.reject_truncation))
    return splits


def collator(tokenizer):
    return lambda rows: collate(rows, tokenizer.pad_token_id)
