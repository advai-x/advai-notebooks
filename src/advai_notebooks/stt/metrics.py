"""Corpus WER/CER: aggregate edit counts, not an average of utterance ratios."""

import unicodedata


def normalize(value, policy):
    if policy == "none":
        return value
    if policy != "nfkc_lower_whitespace":
        raise ValueError("Unknown transcription normalization")
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != candidate))
            )
        previous = current
    return previous[-1]


def error_rates(references, hypotheses, policy="nfkc_lower_whitespace"):
    if not references or len(references) != len(hypotheses):
        raise ValueError("Nonempty paired references and hypotheses are required")
    counts = {
        "word_edits": 0,
        "reference_words": 0,
        "character_edits": 0,
        "reference_characters": 0,
    }
    for reference, hypothesis in zip(references, hypotheses):
        reference, hypothesis = normalize(reference, policy), normalize(hypothesis, policy)
        if not reference.strip():
            raise ValueError("Normalized reference must be nonempty")
        words, predicted_words = reference.split(), hypothesis.split()
        chars, predicted_chars = list("".join(reference.split())), list("".join(hypothesis.split()))
        counts["word_edits"] += edit_distance(words, predicted_words)
        counts["reference_words"] += len(words)
        counts["character_edits"] += edit_distance(chars, predicted_chars)
        counts["reference_characters"] += len(chars)
    return {
        **counts,
        "wer": counts["word_edits"] / counts["reference_words"],
        "cer": counts["character_edits"] / counts["reference_characters"],
    }
