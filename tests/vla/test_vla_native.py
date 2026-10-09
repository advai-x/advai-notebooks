"""Continuous action baselines are normalized, reloadable and horizon-aware."""

from pathlib import Path

import pytest
import torch

from advai_notebooks.vla.config import load_config
from advai_notebooks.vla.data import load_data, prepare_data
from advai_notebooks.vla.model import VLAModel
from advai_notebooks.vla.pipeline import run

ROOT = Path(__file__).resolve().parents[2]


def test_tiny_clip_pools_the_end_of_text_token():
    from advai_notebooks.common.runtime import setup

    cfg = load_config(ROOT / "configs/vla/smoke.yaml")
    model = VLAModel(cfg.model, setup(cfg))
    rows, _ = load_data(cfg)
    inputs = model.inputs(rows[:2])
    end = model.processor.tokenizer.eos_token_id
    assert end == len(model.processor.tokenizer) - 1
    assert end == model.model.encoder.config.text_config.eos_token_id
    with torch.inference_mode():
        result = model.model.encoder.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    positions = (inputs["input_ids"] == end).int().argmax(-1)
    torch.testing.assert_close(
        result.pooler_output, result.last_hidden_state[torch.arange(len(rows[:2])), positions]
    )


@pytest.mark.parametrize("frozen", [True, False])
def test_action_prediction_reload_and_local_clip_finetuning(tmp_path, frozen):
    cfg = load_config(ROOT / "configs/vla/smoke.yaml")
    cfg.model.freeze_backbone = frozen
    cfg.output_dir = str(tmp_path / "run")
    report = run(cfg)
    model = VLAModel.load(report["export"])
    assert all(p.requires_grad != frozen for p in model.model.encoder.parameters())
    rows, _ = load_data(cfg)
    row = rows[-1]
    result = model.predict(row["image"], row["instruction"], row["state"])
    assert torch.tensor(result["actions"]).shape == (cfg.model.horizon, cfg.model.action_dim)
    export = model.save(tmp_path / "export")
    assert result == VLAModel.load(export).predict(row["image"], row["instruction"], row["state"])
    cfg.model.id = str(export / "encoder")
    cfg.output_dir = str(tmp_path / "local-pretrained")
    assert run(cfg)["kind"] == "vla"
    cfg.model.horizon += 1
    with pytest.raises(ValueError, match="dimensions"):
        prepare_data(rows, model, cfg)
