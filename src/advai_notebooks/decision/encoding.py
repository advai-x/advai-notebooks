"""Pair encoding preserves each complete question/option and a common state prefix."""

import json

from .contracts import DecisionRequest, EncodedDecision


def option_texts(question):
    """Compatibility helper for the existing task JSON schema."""
    return request_option_texts(DecisionRequest.from_question("placeholder", question))


def request_option_texts(request):
    return [
        f"type: {request.kind}\nquestion: {request.question}\noption: {option.key}\ncriteria: {option.description}"
        for option in request.options
    ]


class PairInputEncoder:
    def __init__(self, tokenizer, max_len, head_max_len, reject_truncation=True):
        self.tokenizer, self.max_len, self.head_max_len = tokenizer, max_len, head_max_len
        self.reject_truncation = reject_truncation

    def __call__(self, request):
        options, truncated = _encode_request(
            self.tokenizer, request, self.max_len, self.head_max_len, self.reject_truncation
        )
        return EncodedDecision(options, truncated)


def build_input_encoder(tokenizer, max_len, head_max_len, reject_truncation=True, name="pair"):
    if name != "pair":
        raise ValueError(f"Unsupported decision input encoder: {name}")
    return PairInputEncoder(tokenizer, max_len, head_max_len, reject_truncation)


def encode_options(tokenizer, state, question, max_len, head_max_len, reject_truncation):
    request = DecisionRequest.from_question(state, question)
    result = build_input_encoder(tokenizer, max_len, head_max_len, reject_truncation)(request)
    return result.options, result.truncated


def _encode_request(tokenizer, request, max_len, head_max_len, reject_truncation):
    state = request.state
    state_text = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Decision pair encoding requires a fast Hugging Face tokenizer")
    backend = tokenizer.backend_tokenizer
    backend.no_truncation()
    backend.no_padding()
    state_encoding = backend.encode(state_text, add_special_tokens=False)
    head_encodings = [
        backend.encode(text, add_special_tokens=False) for text in request_option_texts(request)
    ]
    state_ids = state_encoding.ids
    heads = [encoding.ids for encoding in head_encodings]
    if len({tuple(ids) for ids in heads}) != len(heads):
        raise ValueError(
            "Options collapsed to identical tokens; use a suitable tokenizer or distinct descriptions"
        )
    if any(len(ids) > head_max_len for ids in heads):
        raise ValueError(
            "Question/option exceeds head_max_len; increase it or shorten the question/criteria"
        )
    # Build pairs through the tokenizer so BERT/RoBERTa special-token conventions differ correctly.
    overhead = tokenizer.num_special_tokens_to_add(pair=True)
    room = max_len - max(map(len, heads)) - overhead
    if room < 1:
        raise ValueError("No room for state; increase max_len or shorten question/option")
    truncated = len(state_ids) > room
    if truncated and reject_truncation:
        raise ValueError("State truncated; increase max_len or explicitly allow truncation")
    if truncated:
        state_encoding.truncate(room, direction="right")
    features = []
    for head in head_encodings:
        encoded = backend.post_process(state_encoding, head, add_special_tokens=True)
        item = {"input_ids": encoded.ids, "attention_mask": encoded.attention_mask}
        if "token_type_ids" in tokenizer.model_input_names:
            item["token_type_ids"] = encoded.type_ids
        features.append(item)
    return features, truncated
