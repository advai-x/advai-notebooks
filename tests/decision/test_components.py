"""Replaceable components, label-free inference and supervised ordinal semantics."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

from advai_notebooks.decision.config import TrainingConfig, load_config
from advai_notebooks.decision.contracts import DecisionOption, DecisionRequest
from advai_notebooks.decision.decoding import build_decoder
from advai_notebooks.decision.encoding import build_input_encoder
from advai_notebooks.decision.inference import predict_requests, predict_state
from advai_notebooks.decision.losses import build_objective
from advai_notebooks.decision.model import DecisionModel
from advai_notebooks.decision.runtime import make_batch

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT / "configs/decision/tasks/triage.json").read_text())


def test_label_free_requests_match_task_wrapper_and_decoder():
    bundle = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    requests = [
        DecisionRequest.from_question("charged twice refund today", q)
        for q in TASKS["triage"].values()
    ]
    encoder = build_input_encoder(bundle.tokenizer, 128, 96)
    encoded = [encoder(request) for request in requests]
    items = [
        {"options": item.options, "qtype": request.qtype}
        for request, item in zip(requests, encoded)
    ]
    batch = make_batch(items, bundle.tokenizer.pad_token_id, torch.device("cpu"))
    assert "target" not in batch and batch["option_mask"].sum(-1).tolist() == [3, 3, 2]
    with pytest.raises(ValueError, match="mix labeled and unlabeled"):
        make_batch([items[0], items[1] | {"target": [1.0, 0.0, 0.0]}], 0, "cpu")
    answers = predict_requests(bundle, requests, torch.device("cpu"), min_confidence=1.0)
    wrapped = predict_state(bundle, requests[0].state, TASKS, "triage", torch.device("cpu"), 1.0)
    assert answers == list(wrapped["answers"].values())
    assert all(answer["abstained"] for answer in answers)
    # Result interpretation is usable without a model or task-definition file.
    score = build_decoder()(requests[1], [0.1, 0.3, 0.6])
    assert score["score"] == pytest.approx(1.5) and score["level"] == 2
    noul = build_decoder()(requests[2], [0.25, 0.75])
    assert noul["noul"] == 0.75


@pytest.mark.parametrize(
    "kind,keys", [("score", ["1", "0"]), ("noul", ["true", "false"]), ("choice", ["a", "a"])]
)
def test_request_rejects_ambiguous_option_contracts(kind, keys):
    with pytest.raises(ValueError):
        DecisionRequest("state", "question?", tuple(DecisionOption(key) for key in keys), kind)


def test_request_rejects_nonfinite_structured_state():
    with pytest.raises(ValueError, match="finite"):
        DecisionRequest(
            {"score": float("nan")}, "question?", (DecisionOption("a"), DecisionOption("b"))
        )


def test_supervised_ordinal_term_penalizes_farther_errors_at_equal_ce():
    settings = TrainingConfig(loss="supervised", ordinal_weight=1.0)
    mask, target = torch.ones(1, 3, dtype=torch.bool), torch.tensor([[1.0, 0.0, 0.0]])
    near, far = torch.tensor([[0.2, 0.7, 0.1]]).log(), torch.tensor([[0.2, 0.1, 0.7]]).log()
    ce = build_objective(replace(settings, loss="ce"))
    objective = build_objective(settings)
    ordinal = torch.tensor([1])
    assert ce(near, target, mask, ordinal) == pytest.approx(float(ce(far, target, mask, ordinal)))
    assert objective(near, target, mask, ordinal) < objective(far, target, mask, ordinal)
    for kind in (0, 2):
        assert objective(near, target, mask, torch.tensor([kind])) == ce(
            near, target, mask, torch.tensor([kind])
        )


def test_mixed_supervised_batch_preserves_example_weighting_and_masked_gradients():
    settings = TrainingConfig()
    target = torch.tensor([[0.0, 1.0, 0.0], [0.2, 0.5, 0.3], [1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True], [True, True, False]])
    qtype = torch.tensor([0, 1, 2])
    logits = torch.randn(3, 3, requires_grad=True)
    objective = build_objective(settings)
    loss = objective(logits, target, mask, qtype)
    expected = (
        sum(
            objective(logits[i : i + 1], target[i : i + 1], mask[i : i + 1], qtype[i : i + 1])
            for i in range(3)
        )
        / 3
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.equal(logits.grad[~mask], torch.zeros(2))


@pytest.mark.parametrize(
    "section,setting,value",
    [
        ("model", "encoding", "unknown"),
        ("model", "head", "interaction"),
        ("model", "decoder", "unknown"),
        ("training", "ordinal_weight", -1),
        ("training", "ordinal_weight", float("nan")),
        ("training", "ordinal_weight", True),
        ("training", "ce_weight", 0),
    ],
)
def test_component_configuration_rejects_unsupported_or_invalid_values(
    tmp_path, section, setting, value
):
    raw = yaml.safe_load((ROOT / "configs/decision/smoke.yaml").read_text())
    raw[section][setting] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_composition_accepts_custom_backbone_and_head_with_gradients():
    class Backbone(nn.Module):
        output_size = 1

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, ids, attention, mask, types=None):
            return (ids.float() * attention).sum(-1, keepdim=True) * self.scale

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.0))

        def forward(self, features, mask, qtype):
            return features.squeeze(-1) * self.scale

    model = DecisionModel(backbone=Backbone(), head=Head())
    ids, mask = torch.tensor([[[1, 2], [3, 4], [0, 0]]]), torch.tensor([[True, True, False]])
    logits = model(ids, torch.ones_like(ids), mask, torch.tensor([0]))
    assert logits.tolist() == [[6.0, 14.0, -10000.0]]
    logits[mask].sum().backward()
    assert model.backbone.scale.grad == 20 and model.head.scale.grad == 10


def test_native_v1_inference_export_migrates_without_probability_changes(tmp_path):
    bundle = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    expected = predict_state(bundle, "charged twice", TASKS, "triage", torch.device("cpu"))
    path = bundle.save(tmp_path / "legacy_native")
    weights = load_file(str(path / "model.safetensors"))
    old_weights = {}
    for key, value in weights.items():
        for new, old in (
            ("backbone.encoder.", "encoder."),
            ("head.type_embedding.", "type_embedding."),
            ("head.scorer.", "head."),
        ):
            if key.startswith(new):
                old_weights[old + key[len(new) :]] = value
                break
    assert len(old_weights) == len(weights)
    save_file(old_weights, str(path / "model.safetensors"))
    config = json.loads((path / "decision_config.json").read_text())
    config["schema_version"] = 1
    for key in ("encoding", "head", "decoder"):
        del config[key]
    (path / "decision_config.json").write_text(json.dumps(config))
    migrated = DecisionModel.load(path)
    assert (
        predict_state(migrated, "charged twice", TASKS, "triage", torch.device("cpu")) == expected
    )
    saved = migrated.save(tmp_path / "v2")
    assert json.loads((saved / "decision_config.json").read_text())["schema_version"] == 2
    assert (
        predict_state(
            DecisionModel.load(saved), "charged twice", TASKS, "triage", torch.device("cpu")
        )
        == expected
    )
