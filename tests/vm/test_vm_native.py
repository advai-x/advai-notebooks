"""World models predict numerical future states and support native fine-tuning."""

from pathlib import Path

import pytest

from advai_notebooks.vm.config import load_config
from advai_notebooks.vm.model import VMModel
from advai_notebooks.vm.pipeline import run

ROOT = Path(__file__).resolve().parents[2]


def test_state_rollout_native_roundtrip_and_finetune(tmp_path):
    cfg = load_config(ROOT / "configs/vm/smoke.yaml")
    cfg.output_dir = str(tmp_path / "base")
    report = run(cfg)
    model = VMModel.load(report["export"])
    result = model.rollout([0, 0], [[0.1, 0], [0, 0.1]])
    assert len(result["observations"]) == 2
    assert model.predict([0, 0], [0.1, 0]) == result["observations"][0]
    assert result == VMModel.load(model.save(tmp_path / "copy")).rollout(
        [0, 0], [[0.1, 0], [0, 0.1]]
    )
    cfg.model.id = report["export"]
    cfg.output_dir = str(tmp_path / "finetune")
    tuned = run(cfg)
    assert VMModel.load(tuned["export"]).statistics == model.statistics
    with pytest.raises(ValueError, match="horizon"):
        model.rollout([0, 0], [[0.1, 0]] * 4)
    with pytest.raises(ValueError, match="finite"):
        model.predict([float("nan"), 0], [0.1, 0])
