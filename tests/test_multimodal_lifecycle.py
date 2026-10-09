"""Native multimodal/state-space tasks have independent, reproducible lifecycles."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"vlm": "VLMModel", "vla": "VLAModel", "vm": "VMModel"}


def modules(kind):
    return {
        name: importlib.import_module(f"advai_notebooks.{kind}.{name}")
        for name in ("config", "model", "pipeline", "data", "losses", "evaluate", "cli", "publish")
    }


def config(kind, output, method="full"):
    cfg = modules(kind)["config"].load_config(ROOT / f"configs/{kind}/smoke.yaml")
    cfg.output_dir = str(output)
    if kind == "vlm":
        cfg.training.method = method
    return cfg


@pytest.mark.parametrize(
    "kind,method", [("vlm", "full"), ("vlm", "lora"), ("vla", "full"), ("vm", "full")]
)
def test_lifecycle_resume_is_exact_and_publish_is_offline(tmp_path, monkeypatch, kind, method):
    task = modules(kind)
    cls = getattr(task["model"], CLASSES[kind])
    uninterrupted = config(kind, tmp_path / "complete", method)
    reference = task["pipeline"].run(uninterrupted)
    interrupted = config(kind, tmp_path / "interrupted", method)

    def stop(epoch, state):
        if epoch == 1:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        task["pipeline"].run(interrupted, callbacks=(stop,))
    resumed = task["pipeline"].run(interrupted, resume=Path(interrupted.output_dir) / "latest")
    assert reference["training"]["history"] == resumed["training"]["history"]
    for export in ("export", "latest_export"):
        a, b = cls.load(reference[export]), cls.load(resumed[export])
        for key, value in a.model.state_dict().items():
            assert torch.equal(value, b.model.state_dict()[key]), key
    completed = task["pipeline"].run(interrupted, resume=Path(interrupted.output_dir) / "latest")
    assert completed["training"] == resumed["training"]
    assert Path(resumed["export"], "advai_metadata.json").is_file()

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda *a, **k: pytest.fail("dry-run must be offline")
    )
    (Path(resumed["export"]) / "training_state.pt").write_bytes(b"must not upload")
    (Path(resumed["export"]) / "raw_data.jsonl").write_text("private")
    plan = task["publish"].publish(resumed["export"], f"test/{kind}", dry_run=True)
    names = {r["path"] for r in plan["files"]}
    assert plan["status"] == "dry_run"
    assert "advai_metadata.json" in names
    assert not {"raw_data.jsonl", "training_state.pt"} & names
    if kind == "vlm":
        assert "chat_template.jinja" in names
    if kind == "vla":
        assert {"encoder/processor_config.json", "action_head.safetensors"} <= names
    interrupted.training.learning_rate *= 2
    with pytest.raises(ValueError, match="mismatch"):
        task["pipeline"].run(interrupted, resume=Path(interrupted.output_dir) / "latest")


@pytest.mark.parametrize("kind", CLASSES)
def test_config_unknown_fields_and_mixed_precision_rejected(tmp_path, kind):
    task = modules(kind)
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  not_a_field: 1\n")
    with pytest.raises(ValueError, match="Unknown"):
        task["config"].load_config(bad, ROOT)
    cfg = config(kind, tmp_path / "out")
    cfg.runtime.precision = "bf16"
    with pytest.raises(ValueError, match="fp32"):
        task["pipeline"].run(cfg)


@pytest.mark.parametrize("kind", CLASSES)
def test_no_other_task_imports_in_cli_and_training(tmp_path, kind):
    code = """
import importlib.abc, importlib, sys
from pathlib import Path
kind, root, output = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
other = set(("vlm", "vla", "vm", "llm", "embedding", "ranker", "decision", "diffusion")) - {kind}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == "advai_notebooks." + k or fullname.startswith("advai_notebooks." + k + ".") for k in other):
            raise AssertionError("Cross-task import: " + fullname)
sys.meta_path.insert(0, Guard())
prefix = "advai_notebooks." + kind
cfg = importlib.import_module(prefix + ".config").load_config(root / "configs" / kind / "smoke.yaml")
cfg.output_dir, cfg.training.epochs = output, 1
result = importlib.import_module(prefix + ".pipeline").run(cfg)
importlib.import_module(prefix + ".cli").main(["validate", "--config", str(root / "configs" / kind / "smoke.yaml")])
print("INDEPENDENT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, kind, str(ROOT), str(tmp_path / kind)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INDEPENDENT_OK" in result.stdout


@pytest.mark.parametrize("kind", CLASSES)
def test_split_and_media_contracts(tmp_path, kind):
    task = modules(kind)
    source = ROOT / "data/sample" / kind
    import shutil

    shutil.copytree(source, tmp_path / "data")
    cfg = config(kind, tmp_path / "out")
    filename = "samples.jsonl" if kind == "vlm" else "episodes.jsonl"
    file = tmp_path / "data" / filename
    cfg.data.samples = str(file)
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    original = copy.deepcopy(rows)
    rows[-1]["group_id"] = rows[0]["group_id"]
    file.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="group"):
        task["data"].load_data(cfg)
    rows = original
    if kind == "vlm":
        rows[-1]["image"] = rows[0]["image"]
    elif kind == "vla":
        rows[-1]["steps"][0]["image"] = rows[0]["steps"][0]["image"]
    else:
        for key in ("observations", "actions", "timestamps"):
            rows[-1][key] = rows[0][key]
    file.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="crosses splits"):
        task["data"].load_data(cfg)


@pytest.mark.parametrize("kind", ("vla", "vm"))
def test_tail_masks_statistics_and_semantics(tmp_path, kind):
    task = modules(kind)
    cfg = config(kind, tmp_path / "out")
    cls = getattr(task["model"], CLASSES[kind])
    from advai_notebooks.common.runtime import setup

    model = cls(cfg.model, setup(cfg), cfg.root, cfg.training)
    rows, _ = task["data"].load_data(cfg)
    prepared = task["data"].prepare_data(rows, model, cfg, fit=True)
    assert prepared["train"][-1]["mask"] == [1] + [0] * (cfg.model.horizon - 1)
    # Masked targets have exactly zero effect on the objective.
    batch = copy.deepcopy(prepared["train"][-1:])
    targets = "actions" if kind == "vla" else "targets"
    for target in batch[0][targets][1:]:
        target[:] = [12345] * len(target)
    before = task["losses"].loss(model, prepared["train"][-1:])
    after = task["losses"].loss(model, batch)
    assert torch.equal(before, after)
    if kind == "vla":
        raw = torch.tensor([r["actions"][0] for r in prepared["train"]])
        assert model.statistics["action_mean"] == raw.mean(0).tolist()
        cfg.data.coordinate_frame = "different"
    else:
        raw = torch.tensor(next(r["_fit_observations"] for r in rows if "_fit_observations" in r))
        assert model.statistics["observation_mean"] == raw.mean(0).tolist()
        cfg.data.observation_units = "different"
    with pytest.raises(ValueError, match="semantics"):
        task["data"].prepare_data(rows, model, cfg)
