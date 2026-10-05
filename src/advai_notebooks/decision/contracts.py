"""Public label-free requests and replaceable decision component contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .schema import QTYPES, option_keys, validate_tasks

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class DecisionOption:
    key: str
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("Option key must be a nonempty string")
        if not isinstance(self.description, str):
            raise ValueError("Option description must be a string")


@dataclass(frozen=True)
class DecisionRequest:
    state: str | dict | list
    question: str
    options: tuple[DecisionOption, ...]
    kind: str = "choice"

    def __post_init__(self):
        if self.kind not in QTYPES:
            raise ValueError("Decision kind must be choice, score or noul")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("Question must be a nonempty string")
        if (
            not isinstance(self.state, (str, dict, list))
            or not self.state
            or (isinstance(self.state, str) and not self.state.strip())
        ):
            raise ValueError("state must be nonempty text, object or list")
        try:
            json.dumps(self.state, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must be JSON serializable with finite values") from exc
        if (
            not isinstance(self.options, (tuple, list))
            or len(self.options) < 2
            or any(not isinstance(option, DecisionOption) for option in self.options)
        ):
            raise ValueError("At least two DecisionOption objects are required")
        object.__setattr__(self, "options", tuple(self.options))
        keys = [option.key for option in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError("Option keys must be distinct")
        if self.kind == "score" and keys != [str(i) for i in range(len(keys))]:
            raise ValueError("Score options must be ordered zero-based levels")
        if self.kind == "noul" and keys != ["false", "true"]:
            raise ValueError("noul options must be false, true in that order")

    @property
    def qtype(self):
        return QTYPES[self.kind]

    @classmethod
    def from_question(cls, state, question):
        validate_tasks({"request": {"question": question}})
        options = []
        criteria = question.get("criteria", {})
        for key in option_keys(question):
            if question["type"] == "score":
                description = criteria[int(key)]
            elif question["type"] == "noul":
                description = question.get("labels", {}).get(key, key) + " " + criteria.get(key, "")
            else:
                description = criteria[key]
            options.append(DecisionOption(key, description))
        return cls(state, question["instructions"], tuple(options), question["type"])


@dataclass
class EncodedDecision:
    options: list[dict[str, list[int]]]
    truncated: bool


class InputEncoder(Protocol):
    def __call__(self, request: DecisionRequest) -> EncodedDecision: ...


class OptionBackbone(Protocol):
    output_size: int

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        option_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        """[B,K,L] tokens -> [B,K,H] pooled option features."""
        ...


class DecisionHead(Protocol):
    def forward(self, features: Tensor, option_mask: Tensor, qtype: Tensor) -> Tensor:
        """[B,K,H] features + [B,K] mask + [B] types -> [B,K] logits."""
        ...


class DecisionObjective(Protocol):
    def __call__(
        self, logits: Tensor, target: Tensor, mask: Tensor, qtype: Tensor, sigma: float = 1.0
    ) -> Tensor:
        """Masked [B,K] logits/targets -> scalar example-averaged loss."""
        ...


class AnswerDecoder(Protocol):
    def __call__(
        self,
        request: DecisionRequest,
        probabilities: list[float],
        min_confidence: float | None = None,
    ) -> dict: ...
