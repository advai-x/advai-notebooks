"""RL checkpoint/RNG transport only; algorithms and checkpoint selection are task-owned."""

import hashlib
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from advai_notebooks.common.artifacts import digest, write_json
from advai_notebooks.llm.runtime import signature


def run_signature(cfg, summary, model):
    # Include initial tensors: changing a local checkpoint under the same path invalidates resume.
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    custom_source = None
    if cfg.reward.custom_function:
        module_name, function_name = cfg.reward.custom_function.split(":")
        module = importlib.import_module(module_name)
        if not callable(getattr(module, function_name, None)):
            raise ValueError("Custom reward must name a callable")
        source_path = getattr(module, "__file__", None)
        if not source_path or not Path(source_path).is_file():
            raise ValueError("Custom reward requires a source file for reproducible recovery")
        custom_source = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    return digest(
        {
            "runtime": signature(cfg, summary, model),
            "initial_weights": hasher.hexdigest(),
            "custom_reward_source": custom_source,
        }
    )


def single_process():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("RANK", "0")) != 0:
        raise ValueError(
            "RL v1 supports a single process/device; distributed launch is unsupported"
        )


def rng_state():
    numpy = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(output, epoch, model, reference, optimizer, state):
    path = Path(output) / "checkpoints" / f"checkpoint-{epoch}"
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise ValueError(
            "Checkpoint already exists; choose a fresh output or remove incomplete checkpoint"
        )
    temporary.mkdir(parents=True)
    model.save(temporary / "policy")
    reference.save(temporary / "reference")
    torch.save({"optimizer": optimizer.state_dict(), "rng": rng_state()}, temporary / "state.pt")
    write_json(temporary / "resume.json", {"schema_version": 1, **state, "epoch": epoch})
    temporary.replace(path)
    write_json(Path(output) / "latest.json", {"checkpoint": str(path.relative_to(output))})
    return path


def resolve_checkpoint(output, resume, expected_signature):
    output = Path(output).resolve()
    path = Path(resume).expanduser().resolve()
    if path == output / "latest":
        pointer = json.loads((output / "latest.json").read_text())
        path = (output / pointer["checkpoint"]).resolve()
    if not path.is_relative_to(output / "checkpoints") or not path.name.startswith("checkpoint-"):
        raise ValueError("Resume must use this experiment's complete RL checkpoint")
    for name in (
        "resume.json",
        "state.pt",
        "policy/advai_metadata.json",
        "reference/advai_metadata.json",
    ):
        if not (path / name).is_file():
            raise ValueError("Incomplete RL checkpoint; actor/reference/optimizer/RNG are required")
    state = json.loads((path / "resume.json").read_text())
    if state.get("schema_version") != 1 or state.get("signature") != expected_signature:
        raise ValueError("Resume configuration/data/source/initial-model signature mismatch")
    tensor_state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    return path, state, tensor_state
