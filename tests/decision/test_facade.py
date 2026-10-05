"""Decision public boundaries aligned with independent embedding/ranker adapters."""

import copy
import json
from pathlib import Path

import pytest
import torch

from advai_notebooks.decision.config import load_config
from advai_notebooks.decision.data import (
    DecisionRequest,
    load_data,
    prepare_items,
    prepare_training_items,
    requests_from_tasks,
)
from advai_notebooks.decision.evaluate import evaluate, evaluate_selected
from advai_notebooks.decision.inference import predict_state
from advai_notebooks.decision.model import DecisionModel
from advai_notebooks.decision.runtime import forward, make_batch

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT / "configs/decision/tasks/triage.json").read_text())


def test_model_load_save_predict_preserves_native_weights_and_raw_forward(tmp_path):
    model = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    path = model.save(tmp_path / "export")
    loaded = DecisionModel.load(path)
    assert loaded.model is loaded and loaded.device == torch.device("cpu")
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    requests = requests_from_tasks("charged twice refund today", TASKS, "triage")
    items = loaded.features(requests)
    assert all("target" not in item for item in items)
    loaded.eval()
    with torch.inference_mode():
        logits = forward(loaded, make_batch(items, loaded.tokenizer.pad_token_id, loaded.device))
    answers = loaded.predict(requests)
    for row, answer, request in zip(logits, answers, requests):
        assert list(answer["probabilities"].values()) == pytest.approx(
            row[: len(request.options)].softmax(-1).tolist(), abs=1e-7
        )
    wrapped = loaded.predict_state(requests[0].state, TASKS, "triage")
    assert list(wrapped["answers"].values()) == answers
    assert loaded.system_one(requests[0].state, task="triage") == wrapped
    assert loaded.system_one(iter(requests)) == answers
    assert loaded.task_definitions == TASKS
    assert (
        DecisionModel.load(loaded.save(tmp_path / "resaved")).system_one(
            requests[0].state, task="triage"
        )
        == wrapped
    )
    assert (
        predict_state(
            DecisionModel.load(path), requests[0].state, TASKS, "triage", torch.device("cpu")
        )
        == wrapped
    )
    assert DecisionModel.load(loaded.save(tmp_path / "resaved")).predict(requests) == answers


def test_system_one_three_typed_outputs_have_correct_values_and_confidence(monkeypatch):
    model = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    probabilities = [[0.2, 0.7, 0.1], [0.1, 0.3, 0.6], [0.25, 0.75]]

    def predict_items(items, batch_size):
        assert batch_size == 2
        assert [item["qtype"] for item in items] == [0, 1, 2]
        assert all("target" not in item for item in items)
        return [{"probabilities": p} for p in probabilities]

    monkeypatch.setattr(model, "predict_items", predict_items)
    result = model.system_one("charged twice", task="triage", batch_size=2, min_confidence=0.65)
    choice, score, noul = result["answers"].values()
    assert choice["type"] == "choice" and choice["choice"] == "technical"
    assert score["type"] == "score" and score["score"] == pytest.approx(1.5)
    assert score["level"] == 2 and score["legend"]["2"] == "high: blocking or time critical"
    assert noul["type"] == "noul" and noul["noul"] == 0.75
    for answer, p in zip(result["answers"].values(), probabilities):
        assert list(answer["probabilities"].values()) == p
        assert answer["confidence"] == answer["answer_confidence"] == max(p)
        assert answer["abstained"] == (max(p) < 0.65)
    requests = requests_from_tasks("charged twice", TASKS, "triage")
    assert model.system_one(requests, batch_size=2, min_confidence=0.65) == list(
        result["answers"].values()
    )


def test_system_one_export_without_tasks_accepts_requests_and_explicit_definitions(tmp_path):
    path = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS).save(
        tmp_path / "legacy-export"
    )
    (path / "tasks.json").unlink()
    model = DecisionModel.load(path)
    requests = requests_from_tasks("charged twice", TASKS, "triage")
    expected = model.system_one(requests)
    with pytest.raises(ValueError, match="no task definitions"):
        model.system_one("charged twice", task="triage")
    assert (
        list(model.system_one("charged twice", task="triage", tasks=TASKS)["answers"].values())
        == expected
    )
    with pytest.raises(ValueError, match="task is required"):
        model.system_one("charged twice", tasks=TASKS)
    with pytest.raises(ValueError, match="Unknown task"):
        model.system_one("charged twice", task="unknown", tasks=TASKS)


def test_data_entry_and_model_features_match_legacy_training_encoding():
    model = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    records, tasks = load_data(
        ROOT / "data/sample/decision/triage.jsonl", ROOT / "configs/decision/tasks/triage.json"
    )
    new = prepare_training_items(records, tasks, model)
    old = prepare_items(records, tasks, model.tokenizer, 128, 96)
    assert new == old
    request = DecisionRequest.from_question("charged twice", tasks["triage"]["department"])
    assert model.predict(iter([request])) == model.predict([request])
    long_request = DecisionRequest.from_question("invoice " * 300, tasks["triage"]["department"])
    with pytest.raises(ValueError, match="truncated"):
        model.check_lengths([long_request])
    assert model.features([long_request], reject_truncation=False)[0]["truncated"]


@pytest.mark.parametrize("enabled", [True, False])
def test_evaluator_owns_metrics_and_fits_only_calibration_split(tmp_path, enabled):
    cfg = load_config(ROOT / "configs/decision/smoke.yaml")
    cfg.calibration.enabled = enabled
    model = DecisionModel.from_config(cfg, TASKS)
    records, tasks = load_data(cfg.path(cfg.data.path), cfg.path(cfg.data.tasks))
    items = prepare_training_items(records, tasks, model)
    data = {
        split: [item for item in items if item["split"] == split]
        for split in ("train", "validation", "calibration", "test")
    }
    path = model.save(tmp_path / "source")
    first = DecisionModel.load(path)
    raw_metrics, raw_predictions = evaluate(first, data, cfg, "test", calibrated=False)
    result = evaluate_selected(first, data, cfg)
    assert result["fine_tuned"] == raw_metrics
    altered = copy.deepcopy(data)
    for item in altered["test"]:
        item["target"] = [1.0, *([0.0] * (len(item["target"]) - 1))]
    other = evaluate_selected(DecisionModel.load(path), altered, cfg)
    assert result["calibration"] == other["calibration"]
    assert [p["probabilities"] for p in result["predictions"]] == [
        p["probabilities"] for p in other["predictions"]
    ]
    if enabled:
        assert result["calibration"]["split"] == "calibration"
        assert result["fine_tuned_calibrated"]["overall"]["n"] == 12
    else:
        first.config.update({"temperature": [3.0] * 3, "temperature_by_options": {"0:3": 9.0}})
        data.pop("calibration")
        result = evaluate_selected(first, data, cfg)
        assert result["calibration"] is None and result["fine_tuned_calibrated"] is None
        assert result["predictions"] == raw_predictions
        assert first.config["temperature"] == [1.0] * 3
        assert "temperature_by_options" not in first.config
