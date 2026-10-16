"""Native sparse CPT/SFT: routing objectives, adapters, recovery and artifacts."""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from test_finetuning import ROOT, facade, logits, modules, weights
from transformers import AutoModelForCausalLM, TrainerCallback

from advai_notebooks.llm.configuration import ModelSettings, TrainingSettings
from advai_notebooks.llm.moe import (
    RoutingAccumulator,
    check_architecture,
    validate_adapter_targets,
    weight_budget,
)
from advai_notebooks.llm.native import tiny_model


def settings(task, path, preset="smoke", family="olmoe"):
    cfg = modules(task)["config"].load_config(ROOT / f"configs/llm/{task}/moe/{preset}.yaml")
    cfg.output_dir = str(path)
    cfg.model.id = "tiny-olmoe" if family == "olmoe" else "tiny-qwen3-moe"
    return cfg


@pytest.mark.parametrize("task", ["base", "instruct"])
@pytest.mark.parametrize("family", ["olmoe", "qwen3_moe"])
@pytest.mark.parametrize("preset", ["smoke", "smoke_lora", "smoke_experts_lora"])
def test_native_moe_lifecycle(tmp_path, task, family, preset, monkeypatch):
    cfg = settings(task, tmp_path / "run", preset, family)
    m, cls = modules(task), facade(task)
    report = m["pipeline"].run(cfg)
    assert report["training"]["completed"]
    audit = report["training"]["moe_gradients"]
    assert audit["optimizer_steps"] == 4
    groups = audit["groups"]
    assert groups["other"]["nonzero_gradient_steps"] == 4
    assert groups["router"]["nonzero_gradient_steps"] == (4 if preset == "smoke" else 0)
    assert groups["experts"]["nonzero_gradient_steps"] == (0 if preset == "smoke_lora" else 4)
    routing = report["finetuned_test"]["moe"]
    assert routing["aux_loss"] > 0
    for layer in routing["layers"]:
        assert sum(layer["assignment_counts"]) == layer["input_tokens"] * 2
        assert sum(layer["assignment_share"]) == pytest.approx(1)
    best, baseline = cls.load(report["export"]), cls.load(tmp_path / "run/baseline")
    latest_path = json.loads((tmp_path / "run/latest.json").read_text())["checkpoint"]
    latest = cls.load(tmp_path / "run" / latest_path, trainable=True)
    assert latest.model.config.model_type == family
    assert latest.model.config.output_router_logits
    assert best.model.config.router_aux_loss_coef == 0.01
    assert best.source["id"] == cfg.model.id
    if preset != "smoke":
        for name, value in weights(baseline).items():
            if "lora_" not in name:
                assert torch.equal(value, weights(latest)[name]), name
        assert all("lora_" in n for n, p in latest.model.named_parameters() if p.requires_grad)
        assert not latest.model.config.advai_moe_train_router
    else:
        assert any(
            not torch.equal(value, weights(latest)[name])
            for name, value in weights(baseline).items()
            if ".mlp.gate." in name
        )
    # Merge supports both module adapters and native 3D expert-parameter adapters.
    merged_path = best.save(tmp_path / "merged", merged=preset != "smoke")
    merged = cls.load(merged_path)
    torch.testing.assert_close(logits(best), logits(merged), rtol=1e-5, atol=1e-5)
    native = AutoModelForCausalLM.from_pretrained(merged_path)
    metadata = json.loads((merged_path / "advai_metadata.json").read_text())
    assert metadata["moe"]["total_parameters"] == sum(p.numel() for p in native.parameters())
    native.eval()
    inputs = best.tokenizer("the sky is blue", return_tensors="pt")
    with torch.no_grad():
        torch.testing.assert_close(logits(best), native(**inputs).logits, rtol=1e-5, atol=1e-5)
    if task == "base":
        assert best.generate(["the sky is"], max_new_tokens=2) == merged.generate(
            ["the sky is"], max_new_tokens=2
        )
    else:
        messages = [{"role": "user", "content": "what color is the sky ?"}]
        assert best.chat(messages, max_new_tokens=2) == merged.chat(messages, max_new_tokens=2)
    import huggingface_hub

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must remain offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    plan = m["publish"].publish(report["export"], "example/moe", dry_run=True)
    assert plan["status"] == "dry_run"
    assert "synthetic" in plan["model_card"].lower()


class StopAfterFirstSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        control.should_training_stop = True


@pytest.mark.parametrize("task", ["base", "instruct"])
@pytest.mark.parametrize("preset", ["smoke", "smoke_experts_lora"])
@pytest.mark.parametrize("family", ["olmoe", "qwen3_moe"])
def test_moe_resume_preserves_weights_history_and_gradient_audit(tmp_path, task, preset, family):
    m, cls = modules(task), facade(task)
    full_cfg = settings(task, tmp_path / "full", preset, family)
    full = m["pipeline"].run(full_cfg)
    split_cfg = settings(task, tmp_path / "split", preset, family)
    first = m["pipeline"].run(split_cfg, callbacks=[StopAfterFirstSave()])
    assert not first["training"]["completed"]
    recovered = m["pipeline"].run(split_cfg, resume=tmp_path / "split/latest")
    assert recovered["training"]["history"] == full["training"]["history"]
    assert recovered["training"]["moe_gradients"] == full["training"]["moe_gradients"]
    assert recovered["finetuned_test"] == full["finetuned_test"]
    a = weights(cls.load(tmp_path / "full/trainer/checkpoint-4"))
    b = weights(cls.load(tmp_path / "split/trainer/checkpoint-4"))
    assert all(torch.equal(t, b[name]) for name, t in a.items())
    assert m["pipeline"].run(split_cfg, resume=tmp_path / "split/latest") == recovered


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_auxiliary_is_target_weighted_and_not_part_of_nll(task):
    losses = modules(task)["losses"]
    torch.manual_seed(17)
    logits_ = torch.randn(2, 5, 13, requires_grad=True)
    labels = torch.tensor([[-100, 2, 4, -100, 3], [-100, -100, 7, -100, -100]])
    window = labels[:, 1:].ne(-100).sum()
    aux = torch.tensor(2.7, requires_grad=True)
    output = SimpleNamespace(logits=logits_, aux_loss=aux)
    lm = losses.loss(output, labels)
    expected = lm + 0.3 * aux
    micro = sum(
        losses.loss(
            SimpleNamespace(logits=logits_[i : i + 1], aux_loss=aux),
            labels[i : i + 1],
            window,
            aux_loss_coef=0.3,
        )
        for i in range(2)
    )
    torch.testing.assert_close(micro, expected)
    total, count = losses.statistics(logits_, labels)
    torch.testing.assert_close(total / count, lm)
    micro.backward()
    assert aux.grad == pytest.approx(0.3)
    with pytest.raises(ValueError, match="finite scalar"):
        losses.loss(SimpleNamespace(logits=logits_, aux_loss=None), labels, aux_loss_coef=0)


def test_router_diagnostics_mask_padding_and_count_all_input_positions():
    model, _ = tiny_model("olmoe")
    accumulator = RoutingAccumulator(model.config)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    output = SimpleNamespace(router_logits=(torch.zeros(6, 4),), aux_loss=torch.tensor(2.0))
    accumulator.update(output, mask, target_count=1)
    report = accumulator.report()
    layer = report["layers"][0]
    assert layer["input_tokens"] == 5  # Not one supervised SFT token, nor six padded tokens.
    assert sum(layer["assignment_counts"]) == 10
    assert layer["mean_router_entropy_nats"] == pytest.approx(1.38629436)


def test_architecture_memory_and_expert_target_guards():
    model, _ = tiny_model("olmoe")
    with pytest.raises(ValueError, match="not RL"):
        check_architecture(model.config, ModelSettings(), supports_moe=False)
    with pytest.raises(ValueError, match="Requested dense"):
        check_architecture(model.config, ModelSettings(architecture="dense"), supports_moe=True)
    with pytest.raises(ValueError, match="supports only"):
        check_architecture(
            SimpleNamespace(model_type="mixtral"), ModelSettings(), supports_moe=True
        )
    with pytest.raises(ValueError, match="Weight-only"):
        weight_budget(model.config, ModelSettings(max_weight_memory_gb=0.000001))
    for target, message in [("missing", "No expert"), ("q_proj.weight", "3D expert")]:
        with pytest.raises(ValueError, match=message):
            validate_adapter_targets(
                model, TrainingSettings(method="lora", lora_target_parameters=[target])
            )
    with pytest.raises(ValueError, match="frozen router"):
        validate_adapter_targets(
            model, TrainingSettings(method="lora", lora_target_modules=["gate"])
        )


def test_real_moe_requires_sha_before_weight_load(tmp_path, monkeypatch):
    import advai_notebooks.llm.native as native

    config = tiny_model("olmoe")[0].config
    monkeypatch.setattr(native.AutoConfig, "from_pretrained", lambda *a, **kw: config)

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before downloading weights")

    monkeypatch.setattr(native.AutoModelForCausalLM, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="immutable"):
        native.initialize(
            ModelSettings(id="example/moe", revision="main"),
            TrainingSettings(),
            tmp_path,
            "cpu",
            supports_moe=True,
        )


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_dtype_and_full_frozen_router_persist(tmp_path, task):
    cfg = settings(task, tmp_path)
    cfg.training.moe_train_router = False
    cfg.model.dtype = "bfloat16"
    model = facade(task)(cfg.model, "cpu", cfg.root, cfg.training)
    assert not any(p.requires_grad for n, p in model.model.named_parameters() if ".mlp.gate." in n)
    exported = model.save(tmp_path / "export")
    again = facade(task).load(exported, trainable=True)
    assert {p.dtype for p in again.model.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for n, p in again.model.named_parameters() if ".mlp.gate." in n)


@pytest.mark.parametrize("task", ["base", "instruct"])
def test_moe_cli_validate_reports_actual_adapter_inventory(tmp_path, task, capsys):
    cli = importlib.import_module(f"advai_notebooks.llm.{task}.cli")
    cli.main(
        ["validate", "--config", str(ROOT / f"configs/llm/{task}/moe/smoke_experts_lora.yaml")]
    )
    # Native SDK progress goes to stderr; the CLI JSON stays machine-readable.
    result = json.loads(capsys.readouterr().out)
    assert result["moe"]["model_type"] == "olmoe"
    assert not result["moe"]["router_trainable"]
    assert result["moe"]["trainable_parameters"] > 0


def test_adapter_owns_policy_even_when_its_base_has_different_defaults(tmp_path):
    cfg = settings("base", tmp_path, "smoke_lora")
    cfg.training.moe_aux_loss_coef = 0.037
    model = facade("base")(cfg.model, "cpu", cfg.root, cfg.training)
    export = model.save(tmp_path / "adapter")
    # Simulate an immutable external base with upstream rather than adapter defaults.
    config_file = export / "base_model/config.json"
    config = json.loads(config_file.read_text())
    config["router_aux_loss_coef"] = 0.001
    config.pop("advai_moe_train_router", None)
    config_file.write_text(json.dumps(config))
    loaded = facade("base").load(export, trainable=True)
    assert loaded.model.config.router_aux_loss_coef == 0.037
    assert not loaded.model.config.advai_moe_train_router
    assert not any(p.requires_grad for n, p in loaded.model.named_parameters() if ".mlp.gate." in n)


@pytest.mark.parametrize("family", ["olmoe", "qwen3_moe"])
def test_expert_only_lora_gradient_checkpointing_and_native_full_handoff(tmp_path, family):
    cfg = settings("base", tmp_path / "base", "smoke_experts_lora", family)
    cfg.training.lora_target_modules = []
    cfg.training.gradient_checkpointing = True
    report = modules("base")["pipeline"].run(cfg)
    groups = report["training"]["moe_gradients"]["groups"]
    assert groups["experts"]["nonzero_gradient_steps"] == 4
    assert groups["other"]["trainable_tensors"] == 0
    assert groups["router"]["trainable_tensors"] == 0
    merged = facade("base").load(report["export"]).save(tmp_path / "merged", merged=True)
    next_cfg = settings("instruct", tmp_path / "instruct", family=family)
    next_cfg.model.id = str(merged)
    # Native files, not a cross-task Python import or another random initialization.
    next_report = modules("instruct")["pipeline"].run(next_cfg)
    assert next_report["training"]["completed"]
    assert next_report["training"]["moe_model"]["router_trainable"]


def test_cuda_weight_lower_bound_fails_without_allocating_weights(monkeypatch):
    config = tiny_model("olmoe")[0].config
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (1, 1000))
    with pytest.raises(ValueError, match="currently free CUDA"):
        weight_budget(config, ModelSettings(max_weight_memory_gb=1), "cuda")
