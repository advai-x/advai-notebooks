"""Document-based CLM data; split before chunking and preserve every target."""

from advai_notebooks.llm.native import collate
from advai_notebooks.llm.records import check_splits, read_records


def load_data(cfg):
    rows = read_records(cfg.path(cfg.data.samples))
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Base documents require nonempty text")
    summary = check_splits(rows, "document_id", lambda row: row["text"])
    return rows, summary


def encode_row(row, model, reject_truncation=False):
    ids = model.tokenizer(row["text"], add_special_tokens=False)["input_ids"]
    eos = model.tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Base CLM requires an EOS token")
    if not ids or ids[-1] != eos:
        ids.append(eos)
    if len(ids) < 2:
        raise ValueError("Document has no next-token target")
    if reject_truncation and len(ids) > model.settings.max_len:
        raise ValueError("Document exceeds max_len; enable loss-preserving document chunking")
    result = []
    # Overlap exactly one context token: boundary targets are neither lost nor duplicated.
    for start in range(0, len(ids) - 1, model.settings.max_len - 1):
        chunk = ids[start : start + model.settings.max_len]
        result.append(
            {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
                "labels": chunk.copy(),
                "sample_id": row["id"],
            }
        )
    return result


def prepare_data(rows, model, cfg):
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        splits[row["split"]].extend(encode_row(row, model, cfg.data.reject_truncation))
    return splits


def collator(tokenizer):
    return lambda rows: collate(rows, tokenizer.pad_token_id)
