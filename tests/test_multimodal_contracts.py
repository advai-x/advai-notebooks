"""Input, weighted-gradient, selection and CLI regressions for new native backends."""

import copy
import importlib
import json
import shutil
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"vlm": "VLMModel", "vla": "VLAModel", "vm": "VMModel"}


def task(kind, tmp_path):
    parts = {
        name: importlib.import_module(f"advai_notebooks.{kind}.{name}")
        for name in ("config", "model", "data", "pipeline", "losses", "evaluate", "cli", "publish")
    }
    cfg = parts["config"].load_config(ROOT / f"configs/{kind}/smoke.yaml")
    cfg.output_dir = str(tmp_path / "run")
    return parts, cfg, getattr(parts["model"], CLASSES[kind])


@pytest.mark.parametrize("kind", CLASSES)
def test_cli_train_evaluate_inference_export_and_preview(tmp_path, capsys, kind):
    parts, cfg, cls = task(kind, tmp_path)
    report = parts["pipeline"].run(cfg)
    model = cls.load(report["export"])
    rows, _ = parts["data"].load_data(cfg)
    row = next(r for r in rows if r["split"] == "test")
    if kind == "vlm":
        command, request = "chat", {"image": row["image"], "prompt": row["prompt"]}
    elif kind == "vla":
        command, request = "predict", {k: row[k] for k in ("image", "instruction", "state")}
    else:
        command, request = (
            "rollout",
            {"observation": row["observation"], "actions": row["actions"][: sum(row["mask"])]},
        )
    inputs, output = tmp_path / "requests.json", tmp_path / "predictions.json"
    inputs.write_text(json.dumps([request]))
    parts["cli"].main(
        [command, "--checkpoint", report["export"], "--input", str(inputs), "--output", str(output)]
    )
    assert len(json.loads(output.read_text())) == 1
    parts["cli"].main(
        [
            "evaluate",
            "--config",
            str(ROOT / f"configs/{kind}/smoke.yaml"),
            "--checkpoint",
            report["export"],
            "--output",
            str(tmp_path / "metrics.json"),
        ]
    )
    assert (
        json.loads((tmp_path / "metrics.json").read_text())["metrics"]
        == report["finetuned_test"]["metrics"]
    )
    parts["cli"].main(
        ["export", "--checkpoint", report["export"], "--output", str(tmp_path / "copied")]
    )
    assert set(cls.load(tmp_path / "copied").model.state_dict()) == set(model.model.state_dict())
    with pytest.raises(ValueError, match="new or empty"):
        parts["cli"].main(
            ["export", "--checkpoint", report["export"], "--output", str(tmp_path / "copied")]
        )
    parts["cli"].main(
        [
            "publish",
            "--export-dir",
            report["export"],
            "--repo-id",
            f"test/{kind}",
            "--dry-run",
            "--output",
            str(tmp_path / "plan.json"),
        ]
    )
    assert json.loads((tmp_path / "plan.json").read_text())["status"] == "dry_run"
    assert "dry_run" in capsys.readouterr().out


@pytest.mark.parametrize("kind", CLASSES)
def test_gradient_accumulation_uses_valid_target_weights(tmp_path, kind):
    from advai_notebooks.common.native_training import train_epoch
    from advai_notebooks.common.runtime import setup

    parts, cfg, cls = task(kind, tmp_path)
    model = cls(cfg.model, setup(cfg), cfg.root, cfg.training)
    rows, _ = parts["data"].load_data(cfg)
    if kind == "vlm":
        rows[0]["answer"] += " green blue"
    prepared = parts["data"].prepare_data(rows, model, cfg, fit=True)
    samples = [prepared["train"][0], prepared["train"][2]]
    assert model.loss_weight(samples[:1]) != model.loss_weight(samples[1:])
    accumulated = copy.deepcopy(model)
    for current, batch_size, accumulation in ((model, 2, 1), (accumulated, 1, 2)):
        cfg.training.batch_size, cfg.training.gradient_accumulation = batch_size, accumulation
        optimizer = torch.optim.SGD(
            [p for p in current.model.parameters() if p.requires_grad], lr=0.01
        )
        torch.manual_seed(123)
        result = train_epoch(current, samples, cfg, optimizer, parts["losses"].loss)
        assert result["optimizer_steps"] == 1
        assert result["n"] == current.loss_weight(samples)
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, accumulated.model.state_dict()[key], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", CLASSES)
def test_bad_media_vectors_and_time_alignment_rejected(tmp_path, kind):
    parts, cfg, _ = task(kind, tmp_path)
    source = ROOT / "data/sample" / kind
    shutil.copytree(source, tmp_path / "data")
    filename = "samples.jsonl" if kind == "vlm" else "episodes.jsonl"
    file = tmp_path / "data" / filename
    cfg.data.samples = str(file)
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    if kind == "vlm":
        rows[0]["image"] = "images/missing.png"
    elif kind == "vla":
        rows[0]["steps"][1]["timestamp"] = 0.11
    else:
        rows[0]["timestamps"][1] = 0.11
    file.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises((ValueError, FileNotFoundError)):
        parts["data"].load_data(cfg)
    if kind == "vla":
        rows[0]["steps"][1]["timestamp"] = 0.1
        rows[0]["steps"][0]["action"] = [float("nan"), 0]
    elif kind == "vm":
        rows[0]["timestamps"][1] = 0.1
        rows[0]["observations"][0] = [0]
    else:
        rows[0]["image"] = "../outside.png"
    file.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        parts["data"].load_data(cfg)


@pytest.mark.parametrize("kind", CLASSES)
def test_validation_includes_baseline_and_test_never_selects(tmp_path, monkeypatch, kind):
    parts, cfg, cls = task(kind, tmp_path)
    original = parts["evaluate"].evaluate
    validation_calls = []

    def forced_metric(model, prepared, cfg, split):
        result = original(model, prepared, cfg, split)
        value = len(validation_calls) if split == "validation" else -999
        result["metrics"][cfg.evaluation.selection_metric] = value
        if split == "validation":
            validation_calls.append(value)
        return result

    monkeypatch.setattr(parts["pipeline"], "evaluate", forced_metric)
    result = parts["pipeline"].run(cfg)
    assert result["training"]["best_epoch"] == 0
    assert result["training"]["best_metric"] == 0
    assert len(validation_calls) == cfg.training.epochs + 1
    baseline, best = cls.load(Path(cfg.output_dir) / "baseline"), cls.load(result["export"])
    assert all(
        torch.equal(v, best.model.state_dict()[k]) for k, v in baseline.model.state_dict().items()
    )


@pytest.mark.parametrize("kind", CLASSES)
def test_invalid_export_cannot_trigger_hub_access(tmp_path, monkeypatch, kind):
    parts, _, _ = task(kind, tmp_path)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: pytest.fail("No network"))
    with pytest.raises(FileNotFoundError, match="local inference export"):
        parts["publish"].publish("missing/model", f"test/{kind}", dry_run=True)
