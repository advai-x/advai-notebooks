"""Decode calibrated distributions independently of encoding and model architecture."""

import math


def validate_confidence(value):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1
    ):
        raise ValueError("min_confidence must be in [0, 1]")


class TypedAnswerDecoder:
    def __call__(self, request, probabilities, min_confidence=None):
        validate_confidence(min_confidence)
        keys = [option.key for option in request.options]
        if (
            len(probabilities) != len(keys)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)
        ):
            raise ValueError("Invalid decision probabilities")
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        answer = {
            "type": request.kind,
            "probabilities": dict(zip(keys, probabilities)),
            "confidence": max(probabilities),
            "answer_confidence": max(probabilities),
        }
        if request.kind == "choice":
            answer["choice"] = keys[index]
        elif request.kind == "score":
            answer.update(
                {
                    "score": sum(i * p for i, p in enumerate(probabilities)),
                    "level": index,
                    "legend": {option.key: option.description for option in request.options},
                }
            )
        else:
            answer["noul"] = probabilities[1]
        if min_confidence is not None:
            answer["abstained"] = max(probabilities) < min_confidence
        return answer


def build_decoder(name="typed"):
    if name != "typed":
        raise ValueError(f"Unsupported answer decoder: {name}")
    return TypedAnswerDecoder()
