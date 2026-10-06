"""Offline publication and Hub-snapshot round trips with real native model exports."""

import hashlib
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pytest
import torch

from advai_notebooks.common.hub import collect_files, resolve_export

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=("embedding", "ranker", "decision"))
def artifact(tmp_path, request):
    kind = request.param
    package = f"advai_notebooks.{kind}"
    cfg = importlib.import_module(package + ".config").load_config(
        ROOT / f"configs/{kind}/smoke.yaml"
    )
    torch.set_num_threads(2)
    if kind == "decision":
        cls = importlib.import_module(package + ".model").DecisionModel
        tasks = json.loads((ROOT / "configs/decision/tasks/triage.json").read_text())
        model = cls.from_config(cfg, tasks)
        model.config.update(
            {"temperature": [1.5, 2.5, 3.5], "temperature_by_options": {"0:3": 2.0}}
        )
        model.source["model_id"] = "/home/local/private-base-model"
    else:
        cls = getattr(
            importlib.import_module(package + ".model"),
            "EmbeddingModel" if kind == "embedding" else "RankerModel",
        )
        cfg.model.query_prefix = "query: "
        cfg.model.document_prefix = "document: "
        model = cls(cfg.model, "cpu")
        model.settings.id = "/home/local/private-base-model"
    export = tmp_path / "export"
    model.save(export)
    if kind == "decision":
        (export / "calibration.json").write_text(json.dumps({"enabled": True}))
    for name in ("training_state.pt", "training_samples.json", "candidates.json", ".env"):
        (export / name).write_text("do not publish raw data or credentials")
    return kind, cls, export, importlib.import_module(package + ".publish").publish


def predict(kind, model):
    if kind == "embedding":
        return model.encode(["reset password"], "query").numpy()
    if kind == "ranker":
        return np.array(model.score([("reset password", "account login password reset")]))
    return model.system_one("charged twice", task="triage")


def test_publication_and_remote_reload_preserve_predictions(artifact, tmp_path, monkeypatch):
    kind, cls, export, publish = artifact
    expected = predict(kind, cls.load(export))
    remote = tmp_path / "remote"
    calls = []

    class Api:
        def __init__(self, token):
            assert token == "test-token"

        def create_repo(self, **kwargs):
            assert kwargs == {
                "repo_id": f"test/{kind}",
                "repo_type": "model",
                "private": True,
                "exist_ok": True,
            }
            calls.append("create")

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=True)

        def upload_folder(self, **kwargs):
            assert kwargs["revision"] == "main"
            assert kwargs["repo_type"] == "model"
            shutil.copytree(kwargs["folder_path"], remote)
            manifest = json.loads((remote / "publish_manifest.json").read_text())
            assert manifest["task"] == kind
            assert set(kwargs["allow_patterns"]) == {
                *[row["path"] for row in manifest["files"]],
                "publish_manifest.json",
            }
            for row in manifest["files"]:
                data = (remote / row["path"]).read_bytes()
                assert row["sha256"] == hashlib.sha256(data).hexdigest()
            assert not any(
                (remote / name).exists()
                for name in (
                    "training_state.pt",
                    "training_samples.json",
                    "candidates.json",
                    ".env",
                )
            )
            for file in remote.rglob("*.json"):
                assert "/home/local/private-base-model" not in file.read_text()
            calls.append("upload")
            return SimpleNamespace(oid="a" * 40, commit_url="https://huggingface.co/test/commit")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    result = publish(export, f"test/{kind}", token="test-token")
    assert result["status"] == "published" and result["commit"] == "a" * 40
    assert calls == ["create", "upload"]
    assert "test-token" not in json.dumps(result)
    assert kind in (remote / "README.md").read_text()

    def download(**kwargs):
        assert kwargs == {
            "repo_id": f"test/{kind}",
            "repo_type": "model",
            "revision": "a" * 40,
            "token": "test-token",
            "cache_dir": tmp_path / "cache",
            "local_files_only": True,
        }
        return str(remote)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    reloaded = cls.load(
        f"test/{kind}",
        revision=result["commit"],
        token="test-token",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )
    actual = predict(kind, reloaded)
    if kind == "decision":
        assert actual == expected
        assert reloaded.config["temperature"] == [1.5, 2.5, 3.5]
        assert reloaded.config["temperature_by_options"] == {"0:3": 2.0}
        assert (remote / "tasks.json").is_file() and (remote / "calibration.json").is_file()
    else:
        np.testing.assert_array_equal(actual, expected)


def test_dry_run_cli_is_offline_and_does_not_modify_export(artifact, tmp_path, monkeypatch, capsys):
    kind, _, export, _ = artifact
    before = {p.relative_to(export): p.read_bytes() for p in export.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline preview must not access Hub or credentials")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    output = tmp_path / "plan.json"
    importlib.import_module(f"advai_notebooks.{kind}.cli").main(
        [
            "publish",
            "--export-dir",
            str(export),
            "--repo-id",
            f"test/{kind}",
            "--dry-run",
            "--public",
            "--license",
            "apache-2.0",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run" and result["private"] is False
    assert "license: apache-2.0" in result["model_card"]
    assert "training_samples.json" not in [row["path"] for row in result["files"]]
    assert before == {
        p.relative_to(export): p.read_bytes() for p in export.rglob("*") if p.is_file()
    }


def test_visibility_mismatch_aborts_upload(artifact, monkeypatch):
    _, _, export, publish = artifact

    class Api:
        def __init__(self, token):
            pass

        def create_repo(self, **kwargs):
            pass

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=False)

        def upload_folder(self, **kwargs):
            raise AssertionError("Visibility mismatch must fail before uploading")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    with pytest.raises(ValueError, match="visibility"):
        publish(export, "test/model")


def test_upload_failure_does_not_write_success_receipt(artifact, tmp_path, monkeypatch):
    kind, _, export, _ = artifact

    class Api:
        def __init__(self, token):
            pass

        def create_repo(self, **kwargs):
            pass

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=True)

        def upload_folder(self, **kwargs):
            raise RuntimeError("Simulated upload failure")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    output = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="upload failure"):
        importlib.import_module(f"advai_notebooks.{kind}.cli").main(
            [
                "publish",
                "--export-dir",
                str(export),
                "--repo-id",
                f"test/{kind}",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_custom_model_card_is_preserved(artifact, tmp_path):
    _, _, export, publish = artifact
    custom = tmp_path / "README.md"
    custom.write_text("---\nlicense: mit\n---\n\n# Domain model\nEvaluation details.\n")
    assert publish(export, "test/model", model_card=custom, dry_run=True)["model_card"] == (
        custom.read_text()
    )


def test_missing_required_weights_fails_before_network_access(artifact, monkeypatch):
    _, _, export, publish = artifact
    (export / "model.safetensors").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete export must fail before accessing Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        publish(export, "test/model")


def test_remote_cli_inference_accepts_pinned_revision(artifact, tmp_path, monkeypatch, capsys):
    kind, cls, export, _ = artifact
    expected = predict(kind, cls.load(export))
    calls = []

    def download(**kwargs):
        assert kwargs["repo_id"] == f"test/{kind}"
        assert kwargs["revision"] == "b" * 40
        assert kwargs["local_files_only"] is True
        assert kwargs["cache_dir"] == str(tmp_path / "cache")
        calls.append(kwargs)
        return str(export)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    flags = ["--revision", "b" * 40, "--local-files-only", "--cache-dir", str(tmp_path / "cache")]
    output = tmp_path / "prediction.json"
    if kind == "decision":
        args = [
            "predict",
            "--model",
            f"test/{kind}",
            "--task",
            "triage",
            "--state",
            "charged twice",
            "--device",
            "cpu",
            *flags,
        ]
    else:
        inputs = tmp_path / "inputs.json"
        if kind == "embedding":
            inputs.write_text(json.dumps(["reset password"]))
            args = ["encode", "--role", "query"]
            output = tmp_path / "vectors.npy"
        else:
            inputs.write_text(json.dumps([["reset password", "account login password reset"]]))
            args = ["rerank"]
        args += [
            "--checkpoint",
            f"test/{kind}",
            "--input",
            str(inputs),
            "--output",
            str(output),
            *flags,
        ]
    importlib.import_module(f"advai_notebooks.{kind}.cli").main(args)
    assert len(calls) == 1
    if kind == "decision":
        assert json.loads(capsys.readouterr().out) == expected
    else:
        actual = (
            np.load(output) if kind == "embedding" else json.loads(output.read_text())["scores"]
        )
        np.testing.assert_array_equal(actual, expected)


def test_card_includes_only_matching_run_metrics(artifact):
    kind, _, export, publish = artifact
    export_key = "export_dir" if kind == "decision" else "export"
    if kind == "decision":
        report = {
            export_key: str(export),
            "best_epoch": 2,
            "fine_tuned_calibrated": {"overall": {"accuracy": 0.8}},
            "fine_tuned": None,
        }
    else:
        report = {
            export_key: str(export),
            "training": {"best_epoch": 0},
            "finetuned_test": {"metrics": {"mrr@3": 0.8}},
        }
        (export.parent / "environment.json").write_text(json.dumps({"tiny_model": True}))
    report["predictions"] = ["private-example-that-must-stay-local"]
    (export.parent / "report.json").write_text(json.dumps(report))
    result = publish(export, "test/model", dry_run=True)
    assert "Held-out test metrics" in result["model_card"]
    assert "0.8" in result["model_card"]
    assert "private-example-that-must-stay-local" not in result["model_card"]
    if kind != "decision":
        assert "Selected checkpoint: epoch 0" in result["model_card"]
        assert "Random tiny model" in result["model_card"]
    report[export_key] = str(export.parent / "other-export")
    (export.parent / "report.json").write_text(json.dumps(report))
    assert "Held-out test metrics" not in publish(export, "test/model", dry_run=True)["model_card"]


def test_local_missing_directory_does_not_trigger_hub_download(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Local path must not trigger a Hub request")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    with pytest.raises(FileNotFoundError):
        resolve_export(tmp_path / "missing")


def test_export_file_selection_rejects_traversal_and_symlinks(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    (root / "model.safetensors").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symbolic"):
        collect_files(root)
    with pytest.raises(ValueError, match="component directory"):
        collect_files(root, directories=("../private",))
