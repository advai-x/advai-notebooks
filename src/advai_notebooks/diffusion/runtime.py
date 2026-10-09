"""Diffusion epoch recovery, source signatures and atomic artifact transport."""

import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import torch

from advai_notebooks.common.artifacts import digest, write_json
from advai_notebooks.common.checkpoint import capture_rng, restore_rng
from advai_notebooks.common.provenance import source_fingerprint


def single_process():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0:
        raise ValueError("Diffusion v1 supports one process/device; distributed launch rejected")


def environment():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "diffusers", "transformers", "peft", "numpy", "Pillow", "safetensors")
    }


def canonical_config(config):
    values = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return {
        k: v
        for k, v in values.items()
        if not k.startswith("_")
        and k not in {"dtype", "torch_dtype", "transformers_version", "architectures"}
    }


def signature(cfg, summary, model):
    hasher = hashlib.sha256()
    for component in ("unet", "vae", "text_encoder"):
        for name, tensor in sorted(getattr(model.pipeline, component).state_dict().items()):
            hasher.update(f"{component}.{name}".encode())
            hasher.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest(
        {
            "config": cfg.to_dict(),
            "data": summary,
            "initial_weights": hasher.hexdigest(),
            "tokenizer": {
                "vocab": model.pipeline.tokenizer.get_vocab(),
                "special_tokens": model.pipeline.tokenizer.special_tokens_map,
                "max_length": model.pipeline.tokenizer.model_max_length,
                "backend": model.pipeline.tokenizer.backend_tokenizer.to_str(),
            },
            "components": {
                name: canonical_config(getattr(model.pipeline, name).config)
                for name in ("unet", "vae", "text_encoder")
            },
            "noise_scheduler": canonical_config(model.noise_scheduler.config),
            "environment": environment(),
            "source": source_fingerprint("diffusion"),
        }
    )


def save_export(model, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    rng = capture_rng(model.device)
    try:
        model.save(temporary)
        if path.exists():
            shutil.rmtree(path)
        temporary.replace(path)
    finally:
        restore_rng(rng, model.device)
    return path


def save_checkpoint(output, epoch, model, optimizer, state):
    path = Path(output) / "checkpoints" / f"checkpoint-{epoch}"
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise ValueError("Checkpoint exists or is incomplete; inspect it before recovery")
    temporary.mkdir(parents=True)
    rng = capture_rng(model.device)
    try:
        model.save(temporary / "model")
        torch.save({"optimizer": optimizer.state_dict(), "rng": rng}, temporary / "state.pt")
        write_json(temporary / "resume.json", {"schema_version": 1, **state, "epoch": epoch})
        temporary.replace(path)
        write_json(Path(output) / "latest.json", {"checkpoint": str(path.relative_to(output))})
    finally:
        restore_rng(rng, model.device)
    return path


def resume_checkpoint(output, resume, expected_signature):
    output = Path(output).resolve()
    path = Path(resume).expanduser().resolve()
    if path == output / "latest":
        path = (output / json.loads((output / "latest.json").read_text())["checkpoint"]).resolve()
    if not path.is_relative_to(output / "checkpoints") or path.parent != output / "checkpoints":
        raise ValueError("Resume requires this experiment's complete checkpoint or latest alias")
    for file in ("resume.json", "state.pt", "model/advai_metadata.json"):
        if not (path / file).is_file():
            raise ValueError("Incomplete checkpoint; inference exports cannot resume training")
    state = json.loads((path / "resume.json").read_text())
    if state.get("schema_version") != 1 or state.get("signature") != expected_signature:
        raise ValueError("Resume configuration/data/source/environment/initial-model mismatch")
    return path, state, torch.load(path / "state.pt", map_location="cpu", weights_only=True)
