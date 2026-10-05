import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from advai_notebooks.decision.config import TrainingConfig, load_config
from advai_notebooks.decision.data import load_data, prepare_items
from advai_notebooks.decision.inference import predict_state
from advai_notebooks.decision.losses import decision_loss
from advai_notebooks.decision.model import DecisionModel
from advai_notebooks.decision.pipeline import run_experiment

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT / "configs/decision/tasks/triage.json").read_text())


def experiment_config(tmp_path, name):
    raw = yaml.safe_load((ROOT / "configs/decision/smoke.yaml").read_text())
    raw["data"]["path"] = str(ROOT / raw["data"]["path"])
    raw["data"]["tasks"] = str(ROOT / raw["data"]["tasks"])
    raw["output_dir"] = str(tmp_path / name)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("kind", ["supervised", "ce", "rlcd"])
def test_losses_have_finite_gradients_and_ignore_padding(kind):
    config = replace(TrainingConfig(), loss=kind)
    target = torch.tensor([[0.2, 0.8, 0.0]])
    mask = torch.tensor([[True, True, False]])
    logits = torch.tensor([[0.5, -0.2, 999.0]], requires_grad=True)
    torch.manual_seed(10)
    loss = decision_loss(logits, target, mask, torch.tensor([0]), config, 0.2)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0
    assert logits.grad[0, :2].abs().sum() > 0
    modified = logits.detach().clone()
    modified[0, 2] = -999.0
    torch.manual_seed(10)
    assert torch.equal(
        loss.detach(), decision_loss(modified, target, mask, torch.tensor([0]), config, 0.2)
    )


def test_tokenization_refuses_dropped_state_and_collapsed_options():
    bundle = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    records, tasks = load_data(
        ROOT / "data/sample/decision/triage.jsonl", ROOT / "configs/decision/tasks/triage.json"
    )
    row = {**records[0], "state": "invoice " * 300}
    bundle.tokenizer.backend_tokenizer.enable_truncation(max_length=5)
    with pytest.raises(ValueError, match="truncated"):
        prepare_items([row], tasks, bundle.tokenizer, 128, 96)
    items = prepare_items([row], tasks, bundle.tokenizer, 128, 96, reject_truncation=False)
    assert all(item["truncated"] for item in items)
    questions = {
        "bad": {
            "q": {
                "type": "choice",
                "instructions": "Which one?",
                "criteria": {"unseenworda": "", "unseenwordb": ""},
            }
        }
    }
    bad = {**records[0], "task": "bad", "gold": {"q": {"choice": "unseenworda"}}}
    with pytest.raises(ValueError, match="collapsed"):
        prepare_items([bad], questions, bundle.tokenizer, 128, 96)


def test_full_pipeline_export_and_native_inference_match(tmp_path):

    path = experiment_config(tmp_path, "complete")
    report = run_experiment(path)
    export = Path(report["export_dir"])
    assert report["best_epoch"] in (0, 1, 2)
    assert report["fine_tuned_calibrated"]["overall"]["n"] == 12
    assert report["calibration"]["samples_by_type"] == {"0": 10, "1": 10, "2": 10}
    assert report["calibration"]["unfitted_types"] == []
    assert (export / "model.safetensors").is_file()
    assert (export / "decision_config.json").is_file()
    bundle = DecisionModel.load(export)
    records, _ = load_data(
        ROOT / "data/sample/decision/triage.jsonl", ROOT / "configs/decision/tasks/triage.json"
    )
    row = next(row for row in records if row["split"] == "test")
    actual = predict_state(bundle, row["state"], TASKS, "triage", torch.device("cpu"))
    predictions = json.loads((export.parent / "predictions.json").read_text())
    for qid, answer in actual["answers"].items():
        expected = next(p for p in predictions if p["id"] == row["id"] and p["qid"] == qid)
        assert list(answer["probabilities"].values()) == pytest.approx(
            expected["probabilities"], abs=1e-7
        )
    assert actual["answers"]["department"]["choice"] in TASKS["triage"]["department"]["criteria"]
    assert 0 <= actual["answers"]["urgency"]["score"] <= 2
    assert 0 <= actual["answers"]["churn_risk"]["noul"] <= 1
    reloaded = DecisionModel.load(bundle.save(tmp_path / "roundtrip"))
    assert predict_state(reloaded, row["state"], TASKS, "triage", torch.device("cpu")) == actual
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(path)


@pytest.mark.parametrize("kind", ["supervised", "rlcd"])
def test_resume_after_epoch_matches_uninterrupted_training(tmp_path, monkeypatch, kind):
    from advai_notebooks.decision import train

    full_path = experiment_config(tmp_path, "uninterrupted")
    path = experiment_config(tmp_path, "resumed")
    for config in (full_path, path):
        raw = yaml.safe_load(config.read_text())
        raw["training"]["loss"] = kind
        config.write_text(yaml.safe_dump(raw))
    uninterrupted = run_experiment(full_path)
    original = train.write_json

    def interrupt_after_first_epoch(artifact, value):
        original(artifact, value)
        if artifact.name == "history.json" and value[-1]["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(train, "write_json", interrupt_after_first_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run_experiment(path)
    # A future inference export may be incomplete; atomic training state still
    # contains the completed epoch's matching model and optimizer weights.
    (tmp_path / "resumed/latest/model.safetensors").write_bytes(b"interrupted export")
    (tmp_path / "resumed/best/model.safetensors").write_bytes(b"interrupted best export")
    resumed = run_experiment(path, resume=tmp_path / "resumed/latest")
    assert resumed["fine_tuned"] == uninterrupted["fine_tuned"]
    assert json.loads((tmp_path / "resumed/history.json").read_text()) == json.loads(
        (tmp_path / "uninterrupted/history.json").read_text()
    )
    states = [
        torch.load(tmp_path / folder / "latest/training_state.pt", weights_only=True)
        for folder in ("uninterrupted", "resumed")
    ]
    for key, value in states[0]["model_weights"].items():
        assert torch.equal(value, states[1]["model_weights"][key]), key
    a, b = (
        DecisionModel.load(resumed["export_dir"]),
        DecisionModel.load(uninterrupted["export_dir"]),
    )
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key]), key
    raw = yaml.safe_load(path.read_text())
    raw["training"]["head_lr"] = 0.123
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="differ"):
        run_experiment(path, resume=tmp_path / "resumed/latest")


def test_explicit_unavailable_device_and_precision_fail_fast():
    from advai_notebooks.decision.runtime import setup

    cfg = load_config(ROOT / "configs/decision/smoke.yaml")
    cfg.runtime.precision = "fp16"
    with pytest.raises(ValueError, match="mixed precision"):
        setup(cfg)
