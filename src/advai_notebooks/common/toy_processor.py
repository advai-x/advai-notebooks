"""Deterministic offline word tokenizer for native multimodal smoke fixtures."""


def tokenizer(*, eos_last=False):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<image>",
        "USER",
        "ASSISTANT",
        ":",
        "what",
        "color",
        "is",
        "the",
        "square",
        "?",
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "purple",
        "cyan",
        "pink",
        "move",
        "left",
        "right",
        "up",
        "down",
        "hold",
        "object",
        "please",
        "describe",
        "this",
        "image",
        ".",
    ]
    if eos_last:
        words.remove("<eos>")
        words.append("<eos>")
    native = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    native.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=native,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    result.add_special_tokens({"additional_special_tokens": ["<image>"]})
    return result
