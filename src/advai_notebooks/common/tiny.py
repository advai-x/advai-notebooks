"""Offline random BERT checkpoints for exercising real training code paths."""

import inspect
import re
from pathlib import Path


def write_tiny_checkpoint(path, max_len, classification=False):
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

    vocabulary = "refund payment invoice card charged twice duplicate app bug crash outage login password account profile change cancellation cancel service plan features pricing subscription billing technical other query document how what can my your please reset update fix explain work soon now settings help error contact email"
    words = sorted(set(re.findall(r"\w+", vocabulary)))
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words) + "\n")
    vocab_key = (
        "vocab" if "vocab" in inspect.signature(BertTokenizerFast).parameters else "vocab_file"
    )
    tokenizer = BertTokenizerFast(**{vocab_key: str(vocab_path)}, do_lower_case=True)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=max_len,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForSequenceClassification(config) if classification else BertModel(config)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path
