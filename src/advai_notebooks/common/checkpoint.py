"""Atomic epoch checkpoints containing aligned model, optimizer and random state."""

import random
from pathlib import Path


def model_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def capture_rng(device):
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "device": torch.cuda.get_rng_state_all()
        if device.type == "cuda"
        else torch.mps.get_rng_state()
        if device.type == "mps"
        else None,
    }


def restore_rng(state, device):
    import numpy as np
    import torch

    random.setstate(state["python"])
    algorithm, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["device"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["device"])


def save_training_state(path, state):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_resume_state(cfg, resume, signature):
    import torch

    path = cfg.path(str(resume))
    if path != cfg.path(cfg.output_dir) / "latest":
        raise ValueError("Resume from this experiment's output_dir/latest")
    if not (path / "training_state.pt").is_file():
        raise ValueError("No complete training state; inference-only checkpoints cannot resume")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1 or state.get("signature") != signature:
        raise ValueError(
            "Resume configuration/data/implementation differ from the saved experiment"
        )
    return state
