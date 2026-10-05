"""Resume equivalence, independent provenance and configurable decision postprocessing."""

import importlib
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from advai_notebooks.common.provenance import source_fingerprint
from advai_notebooks.decision.model import DecisionModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "kind,loss", [("embedding", "mnrl"), ("embedding", "cached_mnrl"), ("ranker", "bce")]
)
def test_retrieval_resume_restores_exact_weights_and_repairs_exports(
    tmp_path, monkeypatch, kind, loss
):
    from advai_notebooks.common import retrieval_training

    load_config = importlib.import_module(f"advai_notebooks.{kind}.config").load_config
    run = importlib.import_module(f"advai_notebooks.{kind}.pipeline").run
    cfg = load_config(ROOT / f"configs/{kind}/smoke.yaml")
    cfg.training.loss = loss
    cfg.training.gradient_accumulation = 3
    if kind == "ranker":
        cfg.training.optimizer = "adam"
    cfg.output_dir = str(tmp_path / "continuous")
    uninterrupted = run(cfg)
    cfg.output_dir = str(tmp_path / "resumed")
    original = retrieval_training.write_json

    def interrupt(path, value):
        original(path, value)
        if path.name == "history.json" and value[-1]["epoch"] == 1:
            raise RuntimeError("interrupted after epoch 1")

    with monkeypatch.context() as patch:
        patch.setattr(retrieval_training, "write_json", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            run(cfg)
    # Simulate a subsequent uncommitted epoch damaging both native exports.
    for directory in ("latest", "best"):
        for file in (tmp_path / "resumed" / directory).rglob("*.safetensors"):
            file.write_bytes(b"interrupted native export")
    resumed = run(cfg, resume=tmp_path / "resumed/latest")
    assert resumed["training"] == uninterrupted["training"]
    assert resumed["finetuned_test"] == uninterrupted["finetuned_test"]
    states = [
        torch.load(tmp_path / folder / "latest/training_state.pt", weights_only=True)
        for folder in ("continuous", "resumed")
    ]
    for field in ("model_weights", "best_model_weights"):
        for key, tensor in states[0][field].items():
            assert torch.equal(tensor, states[1][field][key]), key
    assert torch.equal(states[0]["rng"]["torch"], states[1]["rng"]["torch"])
    cfg.training.learning_rate *= 2
    with pytest.raises(ValueError, match="differ"):
        run(cfg, resume=tmp_path / "resumed/latest")


def test_source_fingerprints_follow_dependencies_instead_of_all_common_files(tmp_path):
    root = tmp_path / "advai_notebooks"
    shutil.copytree(ROOT / "src/advai_notebooks", root)
    kinds = ("embedding", "ranker", "decision")
    original = {kind: source_fingerprint(kind, root) for kind in kinds}
    changes = [
        ("embedding/optimizer.py", {"embedding"}),
        ("common/retrieval_training.py", {"embedding", "ranker"}),
        ("common/runtime.py", set(kinds)),
        ("common/retrieval_config.py", set()),
        ("embedding/metrics.py", {"embedding"}),
        ("decision/metrics.py", {"decision"}),
        ("decision/model.py", {"decision"}),
        ("common/prepared_data.py", {"embedding", "ranker"}),
        ("ranker/metrics.py", {"ranker"}),
        ("embedding/backbones.py", {"embedding"}),
        ("embedding/heads.py", {"embedding"}),
        ("ranker/backbones.py", {"ranker"}),
        ("ranker/heads.py", {"ranker"}),
        ("decision/backbones.py", {"decision"}),
        ("decision/heads.py", {"decision"}),
    ]
    for relative, affected in changes:
        path = root / relative
        source = path.read_text()
        path.write_text(source + "\n# new implementation\n")
        assert {
            kind for kind in kinds if source_fingerprint(kind, root) != original[kind]
        } == affected
        path.write_text(source)


def decision_config(tmp_path, **overrides):
    raw = yaml.safe_load((ROOT / "configs/decision/smoke.yaml").read_text())
    raw["data"]["tasks"] = str(ROOT / raw["data"]["tasks"])
    raw["data"]["path"] = str(ROOT / raw["data"]["path"])
    raw["output_dir"] = str(tmp_path / "decision")
    raw.update(overrides)
    path = tmp_path / "decision.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, raw


def test_decision_can_disable_calibration_without_a_calibration_split(tmp_path):
    from advai_notebooks.decision.pipeline import run_experiment

    path, raw = decision_config(
        tmp_path,
        calibration={"enabled": False},
        evaluation={"selection_metric": "accuracy", "include_baseline": False},
    )
    rows = [json.loads(line) for line in Path(raw["data"]["path"]).read_text().splitlines()]
    data_path = tmp_path / "three_splits.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(row) for row in rows if row["split"] != "calibration")
    )
    raw["data"]["path"] = str(data_path)
    path.write_text(yaml.safe_dump(raw))
    report = run_experiment(path)
    assert report["fine_tuned_calibrated"] is None and report["calibration"] is None
    history = report["training"]["history"]
    assert [row["epoch"] for row in history] == [1, 2]
    best = max(history, key=lambda row: row["validation"]["overall"]["accuracy"])
    assert report["best_epoch"] == best["epoch"]
    bundle = DecisionModel.load(report["export_dir"])
    assert bundle.config["temperature"] == [1.0] * 3
    assert "temperature_by_options" not in bundle.config
    raw["calibration"]["enabled"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="calibration"):
        run_experiment(path)


def test_decision_can_select_baseline_when_training_does_not_improve(tmp_path, monkeypatch):
    from advai_notebooks.decision import train
    from advai_notebooks.decision.pipeline import run_experiment

    path, _ = decision_config(
        tmp_path, evaluation={"selection_metric": "brier", "include_baseline": True}
    )
    original = train.evaluate
    calls = 0

    def validation_with_regression(*args, **kwargs):
        nonlocal calls
        metrics, predictions = original(*args, **kwargs)
        metrics["overall"]["brier"] = float(calls)
        for record in metrics["metric_records"]:
            if record["name"] == "brier" and record["group"] == {"scope": "overall"}:
                record["value"] = float(calls)
        calls += 1
        return metrics, predictions

    monkeypatch.setattr(train, "evaluate", validation_with_regression)
    report = run_experiment(path)
    assert report["best_epoch"] == 0
    assert report["training"]["best_validation_score"] == 0
    assert report["training"]["history"][0]["epoch"] == 0


@pytest.mark.parametrize(
    "section,values",
    [
        ("evaluation", {"selection_metric": "unknown"}),
        ("evaluation", {"include_baseline": "false"}),
        ("calibration", {"enabled": "false"}),
    ],
)
def test_decision_rejects_invalid_lifecycle_settings(tmp_path, section, values):
    from advai_notebooks.decision.config import load_config

    path, _ = decision_config(tmp_path, **{section: values})
    with pytest.raises(ValueError):
        load_config(path)
