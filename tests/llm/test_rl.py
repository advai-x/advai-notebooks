"""Algorithm math, masks, independent lifecycle, portable LoRA and exact recovery."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def modules(kind):
    prefix = f"advai_notebooks.llm.{kind}"
    return SimpleNamespace(
        **{
            name: importlib.import_module(f"{prefix}.{name}")
            for name in (
                "config",
                "data",
                "model",
                "pipeline",
                "reward",
                "losses",
                "train",
                "optimizer",
                "evaluate",
            )
        }
    )


def test_ppo_gae_known_returns_and_padding():
    from advai_notebooks.llm.ppo.losses import advantages_and_returns

    values = torch.tensor([[0.2, 0.4, 100.0], [0.1, 200.0, 300.0]])
    rewards = torch.tensor([[0.0, 1.0, 999.0], [0.5, 999.0, 999.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    advantages, returns = advantages_and_returns(rewards, values, mask, 1, 1, False)
    torch.testing.assert_close(advantages, torch.tensor([[0.8, 0.6, 0.0], [0.4, 0.0, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[1.0, 1.0, 0.0], [0.5, 0.0, 0.0]]))
    normalized, _ = advantages_and_returns(rewards, values, mask)
    assert normalized[mask].mean().item() == pytest.approx(0, abs=1e-6)
    assert normalized[~mask].eq(0).all()


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
def test_clipped_policy_gradient_direction_and_bounds(kind):
    mod = modules(kind)
    settings = mod.config.AlgorithmConfig(kl_coefficient=0.0)
    log_probs = torch.zeros((2, 1), requires_grad=True)
    old = torch.zeros_like(log_probs)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones_like(log_probs, dtype=torch.bool)

    def loss(log):
        if kind == "grpo":
            return mod.losses.objective(log, old, old, advantages, mask, settings)[0]
        zeros = torch.zeros_like(log)
        return mod.losses.objective(
            log, old, advantages[:, None], zeros, zeros, zeros, zeros, mask, settings
        )[0]

    loss(log_probs).backward()
    assert log_probs.grad[0, 0] < 0  # Increase the probability of positive-advantage actions.
    assert log_probs.grad[1, 0] > 0
    extremes = torch.tensor([[2.0], [0.5]]).log().requires_grad_()
    assert loss(extremes).item() == pytest.approx(-0.2)
    loss(extremes).backward()
    assert extremes.grad.eq(0).all()  # Both beneficial extreme ratios are clipped.


def test_grpo_groups_and_zero_variance_and_sequence_weighting():
    mod = modules("grpo")
    advantages = mod.losses.group_advantages(torch.tensor([1.0, 3.0, 5.0, 5.0]), 2)
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        mod.losses.group_advantages(torch.ones(3), 2)
    values = torch.tensor([[1.0, 1.0], [3.0, 999.0]])
    mask = torch.tensor([[True, True], [True, False]])
    assert mod.losses.sequence_mean(values, mask) == 2
    current = torch.tensor([[0.1, 0.2], [-0.3, 99.0]], requires_grad=True)
    loss, stats = mod.losses.objective(
        current,
        current.detach(),
        torch.zeros_like(current),
        torch.zeros(2),
        mask,
        mod.config.AlgorithmConfig(),
    )
    assert stats["reference_kl"] > 0
    loss.backward()
    assert current.grad[1, 1] == 0
    assert current.grad[mask].abs().sum() > 0


def test_ppo_value_clipping_trains_critic():
    mod = modules("ppo")
    values = torch.zeros((1, 1), requires_grad=True)
    zeros = torch.zeros_like(values)
    mask = torch.ones_like(values, dtype=torch.bool)
    loss, stats = mod.losses.objective(
        zeros,
        zeros,
        zeros,
        values,
        zeros,
        torch.ones_like(values),
        zeros,
        mask,
        mod.config.AlgorithmConfig(),
    )
    assert stats["value_loss"] == 0.5
    loss.backward()
    assert values.grad.item() < 0


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
@pytest.mark.parametrize(
    "mutation",
    ["group", "duplicate", "heldout_prompt", "both", "missing_reference", "bad_messages"],
)
def test_data_rejects_leakage_and_malformed_records(tmp_path, kind, mutation):
    mod = modules(kind)
    cfg = mod.config.load_config(ROOT / f"configs/llm/{kind}/smoke.yaml")
    rows = [json.loads(line) for line in cfg.path(cfg.data.samples).read_text().splitlines()]
    if mutation == "group":
        rows[-1]["group_id"] = rows[0]["group_id"]
    elif mutation == "duplicate":
        rows[-1]["id"] = rows[0]["id"]
    elif mutation == "heldout_prompt":
        rows[-1]["messages"] = rows[0]["messages"]
    elif mutation == "both":
        rows[0]["prompt"] = "hello"
    elif mutation == "missing_reference":
        del rows[0]["reference"]
    else:
        rows[0]["messages"] = ["invalid"]
    data = tmp_path / "samples.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows))
    cfg.data.samples = str(data)
    with pytest.raises(ValueError):
        mod.data.load_data(cfg)


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
@pytest.mark.parametrize(
    "setting,value",
    [
        ("runtime.precision", "fp16"),
        ("algorithm.clip_range", 1.0),
        ("algorithm.temperature", float("nan")),
        ("training.lora_dropout", 0.1),
        ("training.gradient_checkpointing", True),
        ("data.reject_truncation", False),
    ],
)
def test_unsupported_or_invalid_configuration(kind, setting, value):
    mod = modules(kind)
    cfg = mod.config.load_config(ROOT / f"configs/llm/{kind}/smoke.yaml")
    section, name = setting.split(".")
    setattr(getattr(cfg, section), name, value)
    with pytest.raises(ValueError):
        mod.config.validate_config(cfg)


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
def test_custom_reward_and_nonfinite_rejection(tmp_path, monkeypatch, kind):
    mod = modules(kind)
    script = tmp_path / "custom_rl_reward.py"
    script.write_text('def reward(row, text):\n    return float(row["metadata"]["value"])\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    settings = mod.config.RewardConfig(
        exact_match_weight=0, custom_function="custom_rl_reward:reward"
    )
    assert mod.reward.score({"metadata": {"value": 2}}, "hello", settings)[0] == 2
    with pytest.raises(ValueError, match="finite"):
        mod.reward.score({"metadata": {"value": float("nan")}}, "hello", settings)
    sys.modules.pop("custom_rl_reward", None)


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.4])
def test_sample_log_probs_match_recomputed_policy_with_left_padding(temperature):
    from advai_notebooks.llm.configuration import ModelSettings
    from advai_notebooks.llm.grpo.model import GRPOModel
    from advai_notebooks.llm.rollout import sample, token_statistics

    torch.set_num_threads(2)
    model = GRPOModel(ModelSettings(max_len=64))
    model.model.train()
    rollout = sample(
        model, ["hello", "what color is the sky ?"], max_new_tokens=5, temperature=temperature
    )
    assert model.model.training
    model.model.eval()
    log_probs, _, _ = token_statistics(model.model, rollout, temperature)
    torch.testing.assert_close(
        log_probs[rollout.completion_mask],
        rollout.old_log_probs[rollout.completion_mask],
        atol=1e-6,
        rtol=1e-6,
    )
    assert not rollout.old_log_probs.requires_grad


def test_eos_is_included_even_when_pad_equals_eos_and_never_masks_real_actions():
    from advai_notebooks.llm.configuration import ModelSettings
    from advai_notebooks.llm.grpo.model import GRPOModel
    from advai_notebooks.llm.rollout import sample

    model = GRPOModel(ModelSettings(max_len=64))
    model.tokenizer.pad_token = model.tokenizer.eos_token
    eos = model.tokenizer.eos_token_id
    calls = 0

    def forward(input_ids, **kwargs):
        nonlocal calls
        logits = torch.full((*input_ids.shape, len(model.tokenizer)), -100.0)
        logits[0, -1, eos] = 100.0
        logits[1, -1, eos if calls == 2 else model.tokenizer.convert_tokens_to_ids("hello")] = 100.0
        calls += 1
        return SimpleNamespace(logits=logits)

    model.model.forward = forward
    rollout = sample(model, ["hello", "hello world"], max_new_tokens=5, greedy=True)
    assert rollout.completion_mask.tolist() == [[True, False, False], [True, True, True]]
    assert rollout.attention_mask[0, -3:].tolist() == [1, 0, 0]


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
@pytest.mark.parametrize("method", ["full", "lora"])
def test_actual_training_checkpoint_export_reload_and_native_logits(tmp_path, kind, method):
    from transformers import AutoModelForCausalLM

    from advai_notebooks.common.runtime import setup
    from advai_notebooks.llm.rollout import forward

    mod = modules(kind)
    cfg = mod.config.load_config(
        ROOT / f"configs/llm/{kind}/{('smoke' if method == 'full' else 'lora')}.yaml"
    )
    cfg.output_dir = str(tmp_path / "run")
    cfg.training.epochs = 1
    cls = getattr(mod.model, "PPOModel" if kind == "ppo" else "GRPOModel")
    initial = cls(cfg.model, setup(cfg), cfg.root, training=cfg.training, generation=cfg.generation)
    initial_weights = {
        name: p.detach().clone() for name, p in initial.model.named_parameters() if p.requires_grad
    }
    report = mod.pipeline.run(cfg)
    trained = cls.load(report["latest_export"], trainable=True)
    assert any(
        not torch.equal(initial_weights[name], p)
        for name, p in trained.model.named_parameters()
        if name in initial_weights
    )
    if kind == "ppo":
        assert trained.value_head.weight.abs().sum() > 0
        assert (Path(report["export"]) / "value_head.safetensors").is_file()
    history = report["training"]["history"]
    assert history[0]["training"]["optimizer_steps"] > 0
    assert history[0]["training"]["rollouts"] == (4 if kind == "ppo" else 16)
    resumed = mod.pipeline.run(cfg, resume=Path(cfg.output_dir) / "latest")
    assert resumed == report
    with pytest.raises(ValueError, match="nonempty"):
        mod.pipeline.run(cfg)
    portable = tmp_path / "portable"
    trained.save(portable)
    loaded = cls.load(portable)
    prompts = ["hello world"]
    assert loaded.generate(prompts, max_new_tokens=3) == trained.generate(prompts, max_new_tokens=3)
    full = portable if method == "full" else trained.save(tmp_path / "merged", merged=True)
    native = AutoModelForCausalLM.from_pretrained(full).eval()
    facade = cls.load(full)
    ids = facade.tokenizer("hello world", return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        torch.testing.assert_close(
            forward(native, ids["input_ids"], ids["attention_mask"]).logits,
            forward(facade.model, ids["input_ids"], ids["attention_mask"]).logits,
        )
    cfg.training.learning_rate *= 2
    with pytest.raises(ValueError, match="signature"):
        mod.pipeline.run(cfg, resume=Path(cfg.output_dir) / "latest")


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
def test_exact_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch, kind):
    mod = modules(kind)
    cfg = mod.config.load_config(ROOT / f"configs/llm/{kind}/smoke.yaml")
    cfg.output_dir = str(tmp_path / "run")
    actual_evaluate = mod.pipeline.evaluate
    validation_calls = 0

    def interrupted(*args, **kwargs):
        nonlocal validation_calls
        if len(args) > 3 and args[3] == "validation":
            validation_calls += 1
            if validation_calls == 3:  # baseline + epoch 1 saved, epoch 2 fails before checkpoint.
                raise RuntimeError("simulated interruption")
        return actual_evaluate(*args, **kwargs)

    monkeypatch.setattr(mod.pipeline, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        mod.pipeline.run(cfg)
    monkeypatch.setattr(mod.pipeline, "evaluate", actual_evaluate)
    resumed = mod.pipeline.run(cfg, resume=Path(cfg.output_dir) / "latest")
    cls = getattr(mod.model, "PPOModel" if kind == "ppo" else "GRPOModel")
    resumed_model = cls.load(resumed["latest_export"])
    cfg.output_dir = str(tmp_path / "uninterrupted")
    uninterrupted = mod.pipeline.run(cfg)
    straight_model = cls.load(uninterrupted["latest_export"])
    assert resumed["training"] == uninterrupted["training"]
    for name, p in resumed_model.model.state_dict().items():
        assert torch.equal(p, straight_model.model.state_dict()[name]), name
    if kind == "ppo":
        assert torch.equal(resumed_model.value_head.weight, straight_model.value_head.weight)


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
def test_task_independence_cli_and_offline_publication(tmp_path, kind):
    script = """
import importlib.abc, importlib, sys
from pathlib import Path
root, output, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
blocked = ["advai_notebooks." + k for k in ("embedding", "ranker", "decision")]
blocked += ["advai_notebooks.llm." + k for k in ("base", "instruct", "ppo", "grpo") if k != kind]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
prefix = "advai_notebooks.llm." + kind
cfg = importlib.import_module(prefix + ".config").load_config(root / f"configs/llm/{kind}/smoke.yaml")
cfg.output_dir, cfg.training.epochs = str(output), 1
report = importlib.import_module(prefix + ".pipeline").run(cfg)
cli = importlib.import_module(prefix + ".cli")
cli.main(["validate", "--config", str(root / f"configs/llm/{kind}/smoke.yaml")])
cli.main(["publish", "--export-dir", report["export"], "--repo-id", "test/" + kind, "--dry-run"])
publication = importlib.import_module(prefix + ".publish").publish(
    report["export"], "test/" + kind, dry_run=True
)
card = publication["model_card"]
assert "proximal policy optimization (PPO)" in card if kind == "ppo" else "group relative policy optimization (GRPO)" in card
assert "Supervised" not in card and "assistant-only next-token labels" not in card
print("RL_INDEPENDENT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), str(tmp_path / kind), kind],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RL_INDEPENDENT_OK" in result.stdout


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
def test_gradient_accumulation_matches_large_batch_including_partial_window(kind):
    from advai_notebooks.common.runtime import setup

    mod = modules(kind)
    cfg = mod.config.load_config(ROOT / f"configs/llm/{kind}/smoke.yaml")
    cfg.training.batch_size = 3
    cfg.algorithm.update_epochs = 1
    cls = getattr(mod.model, "PPOModel" if kind == "ppo" else "GRPOModel")
    results = []
    for accumulated in (False, True):
        model = cls(
            cfg.model, setup(cfg), cfg.root, training=cfg.training, generation=cfg.generation
        )
        reference = model.frozen_reference()
        rows, _ = mod.data.load_data(cfg)
        prepared = mod.data.prepare_data(rows, model, cfg)
        cfg.algorithm.minibatch_size = (2 if kind == "ppo" else 5) if accumulated else 100
        cfg.training.gradient_accumulation = 10 if accumulated else 1
        optimizer = torch.optim.SGD(mod.optimizer.parameters(model), lr=0.01)
        mod.train.train_epoch(model, reference, prepared["train"], cfg, optimizer)
        results.append([p.detach().clone() for p in mod.optimizer.parameters(model)])
    for large, accumulated in zip(*results, strict=True):
        torch.testing.assert_close(large, accumulated, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
def test_reference_parameters_stay_frozen_and_independent(kind):
    from advai_notebooks.common.runtime import setup

    mod = modules(kind)
    cfg = mod.config.load_config(ROOT / f"configs/llm/{kind}/smoke.yaml")
    cls = getattr(mod.model, "PPOModel" if kind == "ppo" else "GRPOModel")
    model = cls(cfg.model, setup(cfg), cfg.root, training=cfg.training, generation=cfg.generation)
    reference = model.frozen_reference()
    before = {key: value.detach().clone() for key, value in reference.model.state_dict().items()}
    assert not any(p.requires_grad for p in reference.model.parameters())
    rows, _ = mod.data.load_data(cfg)
    prepared = mod.data.prepare_data(rows, model, cfg)
    mod.train.train_epoch(
        model,
        reference,
        prepared["train"],
        cfg,
        mod.optimizer.create_optimizer(model, cfg.training),
    )
    assert all(
        torch.equal(value, reference.model.state_dict()[key]) for key, value in before.items()
    )
    if kind == "ppo":
        assert reference.value_head.weight.abs().sum() == 0
        assert model.value_head.weight.abs().sum() > 0


def test_distributed_launch_is_rejected(monkeypatch):
    from advai_notebooks.llm.rl_runtime import single_process

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single process"):
        single_process()


def test_custom_reward_source_edit_changes_resume_signature(tmp_path, monkeypatch):
    from advai_notebooks.common.runtime import setup
    from advai_notebooks.llm.rl_runtime import run_signature

    mod = modules("grpo")
    script = tmp_path / "signature_reward.py"
    script.write_text("def score(row, text):\n    return 1.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = mod.config.load_config(ROOT / "configs/llm/grpo/smoke.yaml")
    cfg.reward.custom_function = "signature_reward:score"
    model = mod.model.GRPOModel(cfg.model, setup(cfg), cfg.root, generation=cfg.generation)
    _, summary = mod.data.load_data(cfg)
    before = run_signature(cfg, summary, model)
    script.write_text("def score(row, text):\n    return 2.0\n")
    assert before != run_signature(cfg, summary, model)
    sys.modules.pop("signature_reward", None)
