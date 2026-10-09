"""Behavioral validation of independent diffusion training, persistence and recovery."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import StableDiffusionPipeline

from advai_notebooks.common.runtime import setup
from advai_notebooks.diffusion import data, losses, pipeline, train
from advai_notebooks.diffusion.config import load_config, validate_config
from advai_notebooks.diffusion.evaluate import evaluate
from advai_notebooks.diffusion.metrics import select_metric
from advai_notebooks.diffusion.model import DiffusionModel
from advai_notebooks.diffusion.publish import publish
from advai_notebooks.diffusion.runtime import signature

ROOT = Path(__file__).resolve().parents[2]


def config(tmp_path, method="full"):
    cfg = load_config(ROOT / f"configs/diffusion/{'lora' if method == 'lora' else 'smoke'}.yaml")
    cfg.output_dir = str(tmp_path / method)
    return cfg


def tensors(model, name):
    return {k: v.detach().clone() for k, v in getattr(model.pipeline, name).state_dict().items()}


def equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("method", ["full", "lora"])
def test_complete_lifecycle_native_reload_freezing_and_publication(tmp_path, method, monkeypatch):
    cfg = config(tmp_path, method)
    report = pipeline.run(cfg)
    baseline = DiffusionModel.load(Path(cfg.output_dir) / "baseline")
    latest = DiffusionModel.load(report["latest_export"])
    selected = DiffusionModel.load(report["export"])
    assert report["training"]["epochs"] == 2
    assert sum(h["training"]["optimizer_steps"] for h in report["training"]["history"]) == 4
    assert not equal(tensors(baseline, "unet"), tensors(latest, "unet"))
    for component in ("vae", "text_encoder"):
        assert equal(tensors(baseline, component), tensors(latest, component))
    if method == "lora":
        for name, tensor in tensors(baseline, "unet").items():
            if "lora_" not in name:
                assert torch.equal(tensor, tensors(latest, "unet")[name]), name
    rows, _ = data.load_data(cfg)
    prepared = data.prepare_data(rows, selected, cfg)
    assert evaluate(selected, prepared, cfg) == report["finetuned_test"]
    before_rng = torch.get_rng_state().clone()
    image = selected.generate(["a red square"], seed=9, num_inference_steps=2)[0]
    assert image.size == (16, 16) and torch.equal(before_rng, torch.get_rng_state())
    native = StableDiffusionPipeline.from_pretrained(
        Path(report["export"]) / "base_model" if method == "lora" else report["export"],
        local_files_only=True,
        use_safetensors=True,
    )
    if method == "lora":
        native.load_lora_weights(report["export"], local_files_only=True)
    native.scheduler = selected.pipeline.scheduler.__class__.from_config(
        selected.pipeline.scheduler.config
    )
    native.set_progress_bar_config(disable=True)
    expected = native(
        ["a red square"],
        height=16,
        width=16,
        guidance_scale=1,
        num_inference_steps=2,
        generator=[torch.Generator().manual_seed(9)],
    ).images[0]
    np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))
    merged = selected.save(tmp_path / "merged", merged=True)
    full = DiffusionModel.load(merged)
    assert full.method == "full"
    # Fusion differs only by floating-point roundoff.
    np.testing.assert_allclose(
        np.asarray(full.generate(["a red square"], seed=9, num_inference_steps=2)[0], dtype=float),
        np.asarray(image, dtype=float),
        atol=1,
    )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda *a, **kw: pytest.fail("Offline preview used Hub")
    )
    preview = publish(report["export"], "test/diffusion", dry_run=True)
    assert preview["status"] == "dry_run" and preview["task"] == "diffusion"
    assert "Random tiny pipeline" in preview["model_card"]
    names = [r["path"] for r in preview["files"]]
    assert "noise_scheduler/scheduler_config.json" in names
    assert not any("state.pt" in n or "images" in n or "checkpoints" in n for n in names)
    if method == "lora":
        assert "pytorch_lora_weights.safetensors" in names
        assert "base_model/model_index.json" in names
    else:
        assert "model_index.json" in names and "unet/diffusion_pytorch_model.safetensors" in names


@pytest.mark.parametrize("method", ["full", "lora"])
def test_interrupted_epoch_resume_is_exact_and_config_changes_are_rejected(tmp_path, method):
    cfg = config(tmp_path / "interrupted", method)

    def stop(epoch, state):
        if epoch == 1:
            raise RuntimeError("simulated interruption after committed epoch")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run(cfg, callbacks=[stop])
    cfg.training.learning_rate *= 2
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.run(cfg, resume=Path(cfg.output_dir) / "latest")
    cfg.training.learning_rate /= 2
    resumed = pipeline.run(cfg, resume=Path(cfg.output_dir) / "latest")
    # A completed resume also repairs/recreates inference exports without overwriting checkpoints.
    assert (
        pipeline.run(cfg, resume=Path(cfg.output_dir) / "latest")["training"] == resumed["training"]
    )
    cfg.output_dir = str(tmp_path / "uninterrupted" / method)
    straight = pipeline.run(cfg)
    assert resumed["training"] == straight["training"]
    a, b = (
        DiffusionModel.load(resumed["latest_export"]),
        DiffusionModel.load(straight["latest_export"]),
    )
    assert equal(tensors(a, "unet"), tensors(b, "unet"))


def test_evaluation_draws_ignore_training_rng_and_batch_size(tmp_path):
    cfg = config(tmp_path)
    model = DiffusionModel(cfg.model, setup(cfg))
    rows, summary = data.load_data(cfg)
    prepared = data.prepare_data(rows, model, cfg)
    original_signature = signature(cfg, summary, model)
    a = evaluate(model, prepared, cfg)
    torch.rand(200)
    cfg.training.batch_size = 7
    b = evaluate(model, prepared, cfg)
    assert a == b
    cfg.training.batch_size = 2
    model.save(tmp_path / "model")
    restored = DiffusionModel.load(tmp_path / "model", trainable=True)
    data.prepare_data(rows, restored, cfg)
    assert signature(cfg, summary, restored) == original_signature
    record = select_metric(a["metrics"], "denoising_mse")
    assert record["direction"] == "minimize" and record["n"] == 2
    with pytest.raises(ValueError, match="context"):
        model.generate(["red " * 100])


def test_noise_targets_and_loss_are_correct():
    from advai_notebooks.diffusion.schedulers import noise_scheduler

    scheduler = noise_scheduler({"num_train_timesteps": 32})
    x, noise, steps = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4), torch.tensor([1, 20])
    assert torch.equal(losses.target(scheduler, x, noise, steps), noise)
    scheduler.register_to_config(prediction_type="v_prediction")
    alphas = scheduler.alphas_cumprod[steps].reshape(-1, 1, 1, 1)
    expected = alphas.sqrt() * noise - (1 - alphas).sqrt() * x
    torch.testing.assert_close(losses.target(scheduler, x, noise, steps), expected)
    torch.testing.assert_close(
        losses.per_image_mse(torch.ones_like(x), torch.zeros_like(x)), torch.ones(2)
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows[4].update(image=rows[0]["image"]), "Duplicate decoded"),
        (lambda rows: rows[4].update(group_id=rows[0]["group_id"]), "group crosses"),
        (lambda rows: rows[4].update(id=rows[0]["id"]), "Duplicate ID"),
        (lambda rows: rows[0].update(image="../outside.png"), "relative"),
    ],
)
def test_data_integrity(tmp_path, mutation, match):
    cfg = config(tmp_path)
    sample = tmp_path / "dataset"
    shutil.copytree(ROOT / "data/sample/diffusion", sample)
    path = sample / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg.data.samples = str(path)
    with pytest.raises(ValueError, match=match):
        data.load_data(cfg)


@pytest.mark.parametrize(
    "field,value", [("method", "dreambooth"), ("epochs", 0), ("gradient_checkpointing", "yes")]
)
def test_unsupported_configuration_is_rejected(tmp_path, field, value):
    cfg = config(tmp_path)
    setattr(cfg.training, field, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_partial_accumulation_weights_examples_not_minibatches(monkeypatch):
    # Two minibatches have sizes 2 and 1; average over examples, not batch means.
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class Unet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = parameter

        def forward(self, noisy, steps, encoder_hidden_states):
            return SimpleNamespace(sample=self.weight.expand_as(noisy))

    model = SimpleNamespace(
        device=torch.device("cpu"),
        settings=SimpleNamespace(resolution=16),
        pipeline=SimpleNamespace(unet=Unet()),
        backbone=SimpleNamespace(
            latents=lambda pixels: pixels, text_features=lambda captions: None
        ),
        noise_scheduler=SimpleNamespace(
            config=SimpleNamespace(prediction_type="epsilon"),
            add_noise=lambda latent, noise, steps: latent,
        ),
    )
    monkeypatch.setattr(train, "batches", lambda rows, size, shuffle: [rows[:2], rows[2:]])
    monkeypatch.setattr(
        train, "pixels", lambda rows, resolution, device: torch.zeros(len(rows), 1, 1, 1)
    )
    monkeypatch.setattr(
        train,
        "noise_and_time",
        lambda x, schedule: (
            torch.full_like(x, 0.0 if len(x) == 2 else -2.0),
            torch.zeros(len(x), dtype=torch.long),
        ),
    )
    cfg = SimpleNamespace(
        training=SimpleNamespace(batch_size=2, gradient_accumulation=2, max_grad_norm=100)
    )
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    train.train_epoch(model, [{"caption": "a"}] * 3, cfg, optimizer)
    assert parameter.item() == pytest.approx(1 - 0.1 * (2 * 2 + 6) / 3)


def test_independence_cli_and_native_base_initialization(tmp_path):
    script = """
import importlib.abc, sys, json
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if any(fullname == name or fullname.startswith(name + ".") for name in
               ["advai_notebooks." + k for k in ("embedding", "ranker", "decision", "llm")]):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from advai_notebooks.diffusion.config import load_config, ModelConfig
from advai_notebooks.diffusion.pipeline import run
from advai_notebooks.diffusion.model import DiffusionModel
from advai_notebooks.diffusion.cli import main
cfg=load_config(root/"configs/diffusion/smoke.yaml");cfg.output_dir=str(output)
cfg.training.epochs=1
report=run(cfg)
# Loading native local components follows the same pretrained pipeline path used for Hub bases.
base=DiffusionModel(ModelConfig(id=report["export"],resolution=16))
assert base.pipeline.unet.config.in_channels == 4
main(["generate","--checkpoint",report["export"],"--input",str(root/"configs/diffusion/prompts.json"),
      "--output",str(output/"cli-images"),"--num-inference-steps","2"])
main(["publish","--export-dir",report["export"],"--repo-id","test/diffusion","--dry-run"])
assert (output/"cli-images/image-0000.png").is_file()
print("DIFFUSION_INDEPENDENT_OK")
"""
    done = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), str(tmp_path / "run")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=90,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "DIFFUSION_INDEPENDENT_OK" in done.stdout


def test_hub_base_is_pinned_and_downloads_only_native_assets(tmp_path, monkeypatch):
    import huggingface_hub

    from advai_notebooks.diffusion.config import ModelConfig

    cfg = config(tmp_path)
    model = DiffusionModel(cfg.model, setup(cfg))
    model.save(tmp_path / "native")
    sha = "a" * 40
    seen = {}

    class API:
        def model_info(self, source, revision):
            assert source == "test/base" and revision == "main"
            return SimpleNamespace(sha=sha)

    def snapshot(source, **kwargs):
        seen.update(kwargs)
        return str(tmp_path / "native")

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    settings = ModelConfig(id="test/base", revision="main", resolution=16)
    loaded = DiffusionModel(settings)
    assert seen["revision"] == sha and loaded.source["revision"] == sha
    assert settings.revision == "main"  # Resolving a revision never mutates the caller's config.
    assert "unet/diffusion_pytorch_model.safetensors" in seen["allow_patterns"]
    assert not any(".bin" in pattern or ".fp16." in pattern for pattern in seen["allow_patterns"])
    with pytest.raises(FileNotFoundError):
        DiffusionModel(ModelConfig(id=str(tmp_path / "missing")))


@pytest.mark.parametrize("method", ["full", "lora"])
def test_checkpointed_velocity_training_and_schedule_roundtrip(tmp_path, method):
    cfg = config(tmp_path, method)
    cfg.training.epochs = 1
    cfg.training.gradient_checkpointing = True
    cfg.scheduler.prediction_type = "v_prediction"
    cfg.scheduler.inference = "ddpm"
    report = pipeline.run(cfg)
    model = DiffusionModel.load(report["latest_export"])
    assert model.noise_scheduler.config.prediction_type == "v_prediction"
    assert model.scheduler_settings.inference == "ddpm"
    assert sum(h["training"]["optimizer_steps"] for h in report["training"]["history"]) == 2
    initial = DiffusionModel.load(Path(cfg.output_dir) / "baseline")
    assert not equal(tensors(initial, "unet"), tensors(model, "unet"))
    generated = model.generate(["a red square"], num_inference_steps=2)[0]
    assert generated.size == (16, 16)
    native = StableDiffusionPipeline.from_pretrained(
        report["latest_export"]
        if method == "full"
        else Path(report["latest_export"]) / "base_model",
        local_files_only=True,
    )
    if method == "lora":
        native.load_lora_weights(report["latest_export"])
        # Native adapters retain the frozen base schedule; deploy the trained schedule too.
        from diffusers import DDPMScheduler

        native.scheduler = DDPMScheduler.from_pretrained(
            Path(report["latest_export"]) / "noise_scheduler", local_files_only=True
        )
    assert native.scheduler.config.prediction_type == "v_prediction"
    native.set_progress_bar_config(disable=True)
    native_image = native(
        ["a red square"],
        height=16,
        width=16,
        num_inference_steps=2,
        guidance_scale=cfg.generation.guidance_scale,
        generator=torch.Generator().manual_seed(cfg.generation.seed),
    ).images[0]
    np.testing.assert_array_equal(np.asarray(generated), np.asarray(native_image))
