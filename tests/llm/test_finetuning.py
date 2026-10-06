"""Behavioral checks for independent CLM/SFT, native exports, LoRA and recovery."""

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, TrainerCallback

ROOT = Path(__file__).resolve().parents[2]


def modules(task):
    prefix = f"advai_notebooks.llm.{task}"
    return {
        name: importlib.import_module(f"{prefix}.{name}")
        for name in ("config", "data", "model", "pipeline", "metrics", "losses", "publish", "cli")
    }


def settings(task, path, method="full"):
    m = modules(task)
    cfg = m["config"].load_config(ROOT / "configs/llm" / task / "smoke.yaml")
    cfg.output_dir = str(path)
    cfg.training.method = method
    cfg.training.lora_rank = 4
    cfg.training.lora_alpha = 8
    cfg.training.lora_target_modules = ["q_proj", "v_proj"]
    if task == "instruct":
        cfg.evaluation.generation_metrics = []
    return cfg


def facade(task):
    return getattr(modules(task)["model"], "BaseModel" if task == "base" else "InstructModel")


def weights(model):
    return {
        name: tensor.detach().cpu().clone() for name, tensor in model.model.state_dict().items()
    }


def logits(model):
    model.model.eval()
    batch = model.tokenizer("the sky is blue", return_tensors="pt")
    with torch.no_grad():
        return model.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits


@pytest.mark.parametrize("task", ["base", "instruct"])
@pytest.mark.parametrize("method", ["full", "lora"])
def test_complete_training_native_reload_merge_and_publication(tmp_path, task, method, monkeypatch):
    m, cls = modules(task), facade(task)
    cfg = settings(task, tmp_path / "run", method)
    report = m["pipeline"].run(cfg)
    assert report["training"]["completed"]
    assert len(report["training"]["history"]) == 2
    assert report["finetuned_test"]["supervised_tokens"] > 0
    assert math.isfinite(report["finetuned_test"]["metrics"]["nll"])
    exported = cls.load(report["export"])
    baseline = cls.load(tmp_path / "run/baseline")
    latest = json.loads((tmp_path / "run/latest.json").read_text())["checkpoint"]
    latest = cls.load(tmp_path / "run" / latest)
    assert any(not torch.equal(t, weights(latest)[name]) for name, t in weights(baseline).items())
    assert exported.backbone.module is exported.model.get_decoder()
    assert exported.head.module is exported.model.get_output_embeddings()
    if task == "base":
        before = exported.generate(["the sky is"], max_new_tokens=3)
    else:
        before = exported.chat(
            [{"role": "user", "content": "what color is the sky ?"}], max_new_tokens=3
        )
    if method == "lora":
        from peft import PeftModel

        assert isinstance(exported.model, PeftModel)
        # All frozen base weights remain identical; only adapter weights were trained.
        for name, value in weights(baseline).items():
            if ".lora_" not in name:
                assert torch.equal(value, weights(latest)[name])
        trainable = cls.load(report["export"], trainable=True)
        assert all(
            "lora_" in name for name, p in trainable.model.named_parameters() if p.requires_grad
        )
        pre_merge = weights(exported)
        merged_path = exported.save(tmp_path / "merged", merged=True)
        merged = cls.load(merged_path)
        assert not isinstance(merged.model, PeftModel)
        torch.testing.assert_close(logits(exported), logits(merged), rtol=1e-5, atol=1e-5)
        assert all(torch.equal(t, weights(exported)[name]) for name, t in pre_merge.items())
        native = AutoModelForCausalLM.from_pretrained(merged_path)
        assert not (merged_path / "adapter_config.json").exists()
    else:
        native = AutoModelForCausalLM.from_pretrained(report["export"])
    native.eval()
    inputs = exported.tokenizer("the sky is blue", return_tensors="pt")
    with torch.no_grad():
        expected = native(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).logits
    torch.testing.assert_close(logits(exported), expected, rtol=1e-5, atol=1e-5)
    relocated = exported.save(tmp_path / "relocated")
    again = cls.load(relocated)
    if task == "base":
        assert again.generate(["the sky is"], max_new_tokens=3) == before
    else:
        assert (
            again.chat([{"role": "user", "content": "what color is the sky ?"}], max_new_tokens=3)
            == before
        )
    # A release package contains inference assets only, including template and adapter base.
    (relocated / "training_state.pt").write_bytes(b"excluded")
    (relocated / "raw-data.jsonl").write_text("excluded")
    (relocated / ".env").write_text("excluded")
    import huggingface_hub

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline publication may not contact the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    plan = m["publish"].publish(relocated, f"tests/llm-{task}", dry_run=True)
    paths = {row["path"] for row in plan["files"]}
    assert not paths & {"training_state.pt", "raw-data.jsonl", ".env"}
    assert "chat_template.jinja" in paths
    assert plan["status"] == "dry_run"
    assert not any("\u4e00" <= c <= "\u9fff" for c in plan["model_card"])
    if method == "lora":
        assert "adapter_model.safetensors" in paths
        assert "base_model/model.safetensors" in paths


class StopAfterFirstSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.epoch >= 1:
            control.should_training_stop = True


@pytest.mark.parametrize("task", ["base", "instruct"])
@pytest.mark.parametrize("method", ["full", "lora"])
def test_interrupted_epoch_recovery_matches_uninterrupted(tmp_path, task, method):
    m, cls = modules(task), facade(task)
    full_cfg = settings(task, tmp_path / "full", method)
    full = m["pipeline"].run(full_cfg)
    split_cfg = settings(task, tmp_path / "split", method)
    first = m["pipeline"].run(split_cfg, callbacks=[StopAfterFirstSave()])
    assert first["training"]["last_epoch"] == 1
    assert not first["training"]["completed"]
    recovered = m["pipeline"].run(split_cfg, resume=tmp_path / "split/latest")
    assert recovered["training"]["completed"]
    assert recovered["training"]["history"] == full["training"]["history"]
    for relative in ("best", "trainer/checkpoint-4"):
        left = weights(cls.load(tmp_path / "full" / relative))
        right = weights(cls.load(tmp_path / "split" / relative))
        assert all(torch.equal(t, right[name]) for name, t in left.items())
    assert recovered["finetuned_test"] == full["finetuned_test"]
    # Completed-run recovery is idempotent and validates the same signature.
    assert m["pipeline"].run(split_cfg, resume=tmp_path / "split/latest") == recovered
    split_cfg.training.learning_rate *= 2
    with pytest.raises(ValueError, match="differs"):
        m["pipeline"].run(split_cfg, resume=tmp_path / "split/latest")
    with pytest.raises(ValueError, match="complete Trainer"):
        m["pipeline"].run(full_cfg, resume=Path(full["export"]))


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_gradient_accumulation_matches_large_batch_with_unequal_lengths_and_tail(tmp_path, task):
    m, cls = modules(task), facade(task)
    cfg = settings(task, tmp_path / "small")
    rows, _ = m["data"].load_data(cfg)
    rows = [row for row in rows if row["id"] != "train-7"]
    data = tmp_path / "samples.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows))
    cfg.data.samples, cfg.training.epochs = str(data), 1
    cfg.training.batch_size, cfg.training.gradient_accumulation = 1, 4
    m["pipeline"].run(cfg)
    cfg.output_dir = str(tmp_path / "large")
    cfg.training.batch_size, cfg.training.gradient_accumulation = 4, 1
    m["pipeline"].run(cfg)
    small = weights(cls.load(tmp_path / "small/trainer/checkpoint-2"))
    large = weights(cls.load(tmp_path / "large/trainer/checkpoint-2"))
    for name, value in small.items():
        torch.testing.assert_close(value, large[name], rtol=1e-4, atol=1e-6)


def test_base_chunking_keeps_boundary_targets_and_eos_when_pad_is_eos():
    m = modules("base")
    model = facade("base")()
    model.settings.max_len = 4
    row = {"id": "doc", "text": "the sky is blue and the grass is green ."}
    chunks = m["data"].encode_row(row, model)
    ids = model.tokenizer(row["text"], add_special_tokens=False)["input_ids"]
    expected = [*ids[1:], model.tokenizer.eos_token_id]
    assert [t for chunk in chunks for t in chunk["labels"][1:]] == expected
    model.tokenizer.pad_token = model.tokenizer.eos_token
    batch = m["data"].collator(model.tokenizer)([chunks[0], chunks[-1]])
    assert batch["labels"][1, len(chunks[-1]["labels"]) - 1] == model.tokenizer.eos_token_id
    assert batch["labels"][1, -1] == -100
    with pytest.raises(ValueError, match="chunking"):
        m["data"].encode_row(row, model, reject_truncation=True)


@pytest.mark.parametrize("generation_blocks", [True, False])
def test_sft_masks_system_user_roles_and_supervises_all_assistant_turns(generation_blocks):
    from advai_notebooks.llm.native import TINY_TEMPLATE

    m = modules("instruct")
    model = facade("instruct")()
    if not generation_blocks:
        model.tokenizer.chat_template = TINY_TEMPLATE.replace("{% generation %}", "").replace(
            "{% endgeneration %}", ""
        )
    messages = [
        {"role": "system", "content": "answer only"},
        {"role": "user", "content": "what color is the sky ?"},
        {"role": "assistant", "content": "blue"},
        {"role": "user", "content": "what color is the grass ?"},
        {"role": "assistant", "content": "green"},
    ]
    row = m["data"].encode_row({"id": "two", "messages": messages}, model)[0]
    targets = [t for t in row["labels"] if t != -100]
    assert model.tokenizer.convert_ids_to_tokens(targets) == ["blue", "<|end|>", "green", "<|end|>"]
    model.settings.max_len = 5
    with pytest.raises(ValueError, match="exceeds"):
        m["data"].encode_row({"id": "two", "messages": messages}, model)
    with pytest.raises(ValueError, match="no supervised"):
        m["data"].encode_row({"id": "two", "messages": messages}, model, False)
    model.tokenizer.chat_template = (
        "{% for m in messages %}{{m['role']}}: {{m['content']}} {% endfor %}"
    )
    with pytest.raises(ValueError, match="generation prefix"):
        m["data"].assistant_encoding(messages, model.tokenizer)


@pytest.mark.parametrize("task", ["base", "instruct"])
@pytest.mark.parametrize("mutation", ["group", "content", "id", "split"])
def test_data_integrity_rejects_cross_split_leakage(tmp_path, task, mutation):
    m, cfg = modules(task), settings(task, tmp_path / "run")
    rows, _ = m["data"].load_data(cfg)
    if mutation == "group":
        key = "document_id" if task == "base" else "conversation_id"
        rows[-1][key] = rows[0][key]
    elif mutation == "content":
        key = "text" if task == "base" else "messages"
        rows[-1][key] = rows[0][key]
    elif mutation == "id":
        rows[-1]["id"] = rows[0]["id"]
    else:
        rows[-1]["split"] = "unknown"
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    cfg.data.samples = str(path)
    with pytest.raises(ValueError):
        m["data"].load_data(cfg)


def test_instruct_prompt_completion_normalization_and_generation_metrics(tmp_path):
    m, cfg = modules("instruct"), settings("instruct", tmp_path / "run")
    rows = [
        {
            "id": s,
            "conversation_id": s,
            "split": s,
            "prompt": f"what color is {s}",
            "completion": "blue",
        }
        for s in ("train", "validation", "test")
    ]
    path = tmp_path / "pc.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    cfg.data.samples = str(path)
    normalized, _ = m["data"].load_data(cfg)
    assert normalized[0]["messages"][-1] == {"role": "assistant", "content": "blue"}
    predictions = [
        {"prediction": '{"ok": true}', "reference": '{"ok": true}'},
        {"prediction": "not JSON", "reference": "blue"},
    ]
    result = m["metrics"].add_generation_metrics(
        m["metrics"].report(6, 3, 2), predictions, ["exact_match", "json_validity"]
    )
    assert result["metrics"] == {
        "nll": 2,
        "perplexity": math.exp(2),
        "exact_match": 0.5,
        "json_validity": 0.5,
    }
    assert result["metric_records"][-1]["n"] == 2


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_evaluation_is_token_weighted_and_restores_training_mode(tmp_path, task):
    m, cfg = modules(task), settings(task, tmp_path / "run")
    cfg.training.batch_size = 1
    torch.set_num_threads(2)
    model = facade(task)()
    rows, _ = m["data"].load_data(cfg)
    prepared = m["data"].prepare_data(rows, model, cfg)
    model.model.train()
    one = m["pipeline"].evaluate(model, prepared, cfg, "test", generate_metrics=False)
    assert model.model.training
    cfg.training.batch_size = 8
    many = m["pipeline"].evaluate(model, prepared, cfg, "test", generate_metrics=False)
    assert one["supervised_tokens"] == many["supervised_tokens"]
    assert one["metrics"]["nll"] == pytest.approx(many["metrics"]["nll"], abs=1e-6)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_independent_package_cli_pipeline_and_publish_without_other_tasks(tmp_path, task):
    code = r"""
import importlib.abc
import importlib
import sys
from pathlib import Path
root, output, task = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
other = "instruct" if task == "base" else "base"
blocked = ["advai_notebooks.embedding", "advai_notebooks.ranker", "advai_notebooks.decision",
           "advai_notebooks.llm." + other]
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + ".") for n in blocked):
            raise AssertionError("Cross-task import: " + fullname)
sys.meta_path.insert(0, Block())
prefix = "advai_notebooks.llm." + task
cfg = importlib.import_module(prefix+".config").load_config(root/"configs/llm"/task/"smoke.yaml")
cfg.output_dir = str(output)
cfg.training.epochs = 1
report = importlib.import_module(prefix+".pipeline").run(cfg)
cli = importlib.import_module(prefix+".cli")
cli.main(["validate", "--config", str(root/"configs/llm"/task/"smoke.yaml")])
cli.main(["publish", "--export-dir", report["export"], "--repo-id", "tests/llm-"+task, "--dry-run"])
print("INDEPENDENCE_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(tmp_path / "run"), task],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INDEPENDENCE_OK" in result.stdout


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_base_export_can_initialize_instruct_without_task_module_dependency(tmp_path, task):
    # Native model weights and tokenizer form the contract between the two tasks.
    base = facade("base")()
    path = base.save(tmp_path / "base-export")
    model = facade(task)(modules(task)["config"].ModelConfig(id=str(path)))
    torch.testing.assert_close(logits(base), logits(model), rtol=0, atol=0)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_cli_inference_evaluation_and_export(tmp_path, task):
    m, cls = modules(task), facade(task)
    model = cls()
    checkpoint = model.save(tmp_path / "export")
    value = ["the sky is"] if task == "base" else [[{"role": "user", "content": "hello"}]]
    inputs = tmp_path / "input.json"
    inputs.write_text(json.dumps(value))
    command = "generate" if task == "base" else "chat"
    output = tmp_path / "result.json"
    m["cli"].main(
        [
            command,
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--max-new-tokens",
            "2",
        ]
    )
    assert len(json.loads(output.read_text())["outputs"]) == 1
    m["cli"].main(
        [
            "evaluate",
            "--config",
            str(ROOT / "configs/llm" / task / "smoke.yaml"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]
    )
    assert json.loads(output.read_text())["split"] == "test"
    m["cli"].main(["export", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "copy")])
    torch.testing.assert_close(logits(model), logits(cls.load(tmp_path / "copy")))
    with pytest.raises(ValueError, match="max_len"):
        from advai_notebooks.llm.native import generate

        generate(model, ["the sky is"], max_new_tokens=128)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_hub_package_upload_and_pinned_snapshot_reload(tmp_path, task, monkeypatch):
    import shutil
    from types import SimpleNamespace

    import huggingface_hub

    m, cls = modules(task), facade(task)
    model = cls()
    export = model.save(tmp_path / "export")
    published = tmp_path / "published"
    seen = {}

    class Api:
        def __init__(self, token=None):
            assert token is None

        def create_repo(self, **kwargs):
            assert kwargs["private"] is True
            seen["repo_id"] = kwargs["repo_id"]

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=True)

        def upload_folder(self, **kwargs):
            shutil.copytree(kwargs["folder_path"], published)
            seen["allow_patterns"] = kwargs["allow_patterns"]
            return SimpleNamespace(
                oid="a" * 40, commit_url="https://huggingface.co/tests/model/commit/" + "a" * 40
            )

    def snapshot(**kwargs):
        assert kwargs["revision"] == "a" * 40
        assert kwargs["local_files_only"] is True
        assert kwargs["repo_id"] == f"tests/{task}"
        return str(published)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    receipt = m["publish"].publish(export, f"tests/{task}")
    assert receipt["status"] == "published"
    assert receipt["commit"] == "a" * 40
    assert set(seen["allow_patterns"]) == {
        *[row["path"] for row in receipt["files"]],
        "publish_manifest.json",
    }
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    downloaded = cls.load(f"tests/{task}", revision=receipt["commit"], local_files_only=True)
    torch.testing.assert_close(logits(model), logits(downloaded), rtol=0, atol=0)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_epoch_zero_can_remain_selected_and_missing_state_cannot_resume(
    tmp_path, task, monkeypatch
):
    m, cls = modules(task), facade(task)
    trainer = importlib.import_module(f"advai_notebooks.llm.{task}.train")
    monkeypatch.setattr(trainer, "improves", lambda *args: False)
    cfg = settings(task, tmp_path / "run")
    cfg.training.epochs = 1
    report = m["pipeline"].run(cfg)
    assert report["training"]["best_epoch"] == 0
    left, right = weights(cls.load(tmp_path / "run/baseline")), weights(cls.load(report["export"]))
    assert all(torch.equal(t, right[name]) for name, t in left.items())
    latest = json.loads((tmp_path / "run/latest.json").read_text())["checkpoint"]
    (tmp_path / "run" / latest / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="lacks complete"):
        m["pipeline"].run(cfg, resume=tmp_path / "run/latest")


@pytest.mark.parametrize("task", ["base", "instruct"])
@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model", "max_len", 1),
        ("runtime", "device", "unknown"),
        ("training", "method", "qlora"),
        ("training", "method", []),
        ("runtime", "device", {}),
        ("training", "lora_dropout", 1),
        ("training", "lora_target_modules", ["q_proj", "q_proj"]),
        ("training", "epochs", True),
        ("generation", "top_p", 2),
    ],
)
def test_configuration_rejects_invalid_training_contract(tmp_path, task, section, field, value):
    import yaml

    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({section: {field: value}}))
    with pytest.raises(ValueError):
        modules(task)["config"].load_config(path, root=ROOT)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_metric_selection_validates_task_direction_and_token_denominator(task):
    import copy

    metrics = modules(task)["metrics"]
    selected = metrics.report(20, 10, 2)
    candidate = metrics.report(10, 10, 2)
    assert metrics.improves(candidate, selected)
    assert not metrics.improves(selected, candidate)
    for field, value in (("direction", "higher"), ("n", 2), ("value", float("nan"))):
        invalid = copy.deepcopy(candidate)
        invalid["metric_records"][0][field] = value
        with pytest.raises(ValueError):
            metrics.improves(invalid, selected)
    invalid = copy.deepcopy(candidate)
    invalid["task"] = "llm/instruct" if task == "base" else "llm/base"
    with pytest.raises(ValueError, match="another task"):
        metrics.improves(invalid, selected)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_lora_gradient_checkpointing_trains_only_adapter_parameters(tmp_path, task):
    m, cls = modules(task), facade(task)
    cfg = settings(task, tmp_path / "run", "lora")
    cfg.training.epochs = 1
    cfg.training.gradient_checkpointing = True
    report = m["pipeline"].run(cfg)
    assert report["training"]["completed"]
    initial = weights(cls.load(tmp_path / "run/baseline"))
    latest = weights(cls.load(tmp_path / "run/trainer/checkpoint-2"))
    assert any(not torch.equal(t, latest[name]) for name, t in initial.items() if "lora_" in name)
    assert all(torch.equal(t, latest[name]) for name, t in initial.items() if "lora_" not in name)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_adapter_only_exports_require_merge_before_new_training(tmp_path, task):
    cfg = settings(task, tmp_path / "unused", "lora")
    model = facade(task)(cfg.model, training=cfg.training)
    adapter = model.save(tmp_path / "adapter")
    with pytest.raises(ValueError, match="full or merged"):
        facade(task)(modules(task)["config"].ModelConfig(id=str(adapter)))


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_quantized_initial_config_is_rejected_before_weight_loading(tmp_path, task):
    from transformers import LlamaConfig

    config = LlamaConfig()
    config.quantization_config = {"load_in_4bit": True, "quant_method": "bitsandbytes"}
    config.save_pretrained(tmp_path / "quantized")
    with pytest.raises(ValueError, match="Quantized initial"):
        facade(task)(modules(task)["config"].ModelConfig(id=str(tmp_path / "quantized")))
