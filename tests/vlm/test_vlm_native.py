"""LLaVA supervision, reload, local pretrained bases and merged adapters."""

import copy
from pathlib import Path

import pytest
import torch

from advai_notebooks.common.runtime import setup
from advai_notebooks.vlm.config import load_config
from advai_notebooks.vlm.data import load_data, prepare_data
from advai_notebooks.vlm.model import VLMModel
from advai_notebooks.vlm.pipeline import run

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("method", ["full", "lora"])
def test_native_supervision_and_model_roundtrip(tmp_path, method):
    cfg = load_config(ROOT / "configs/vlm/smoke.yaml")
    cfg.training.method = method
    model = VLMModel(cfg.model, setup(cfg), cfg.root, cfg.training)
    rows, _ = load_data(cfg)
    prepared = prepare_data(rows, model, cfg)
    encoded = prepared["train"][0]
    prompt = model.encode(rows[0], supervise=False)
    boundary = prompt["input_ids"].shape[1]
    assert (encoded["labels"][:, :boundary] == -100).all()
    assert (encoded["labels"][:, boundary:] != -100).all()
    assert (encoded["input_ids"] == model.model.config.image_token_index).sum() == 4
    export = model.save(tmp_path / "export")
    loaded = VLMModel.load(export)
    request = (rows[-1]["image"], rows[-1]["prompt"])
    assert model.chat(*request) == loaded.chat(*request)
    if method == "lora":
        assert all(
            not p.requires_grad for n, p in model.model.named_parameters() if "lora_" not in n
        )
        merged = VLMModel.load(model.save(tmp_path / "merged", merged=True))
        assert merged.method == "full"
        model.model.eval()
        merged.model.eval()
        with torch.inference_mode():
            a = model.model(**model.batch(prepared["train"][:1])).logits
            b = merged.model(**merged.batch(prepared["train"][:1])).logits
        torch.testing.assert_close(a, b)
    else:
        cfg.model.id = str(export)
        cfg.output_dir = str(tmp_path / "finetune")
        assert run(cfg)["kind"] == "vlm"


def test_context_overflow_and_image_traversal(tmp_path):
    cfg = load_config(ROOT / "configs/vlm/smoke.yaml")
    model = VLMModel(cfg.model, setup(cfg))
    rows, _ = load_data(cfg)
    row = copy.deepcopy(rows[0])
    row["prompt"] = "what " * 200
    with pytest.raises(ValueError, match="context budget"):
        model.encode(row)
    from advai_notebooks.common.media import local_image

    with pytest.raises(ValueError):
        local_image(tmp_path, "../outside.ppm")
