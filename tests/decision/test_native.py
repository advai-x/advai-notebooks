"""Native model loading, probability objectives and task/package isolation."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from advai_notebooks.decision.calibrate import fit_calibration
from advai_notebooks.decision.config import load_config
from advai_notebooks.decision.data import load_data, prepare_items
from advai_notebooks.decision.inference import predict_state
from advai_notebooks.decision.losses import proper_reward
from advai_notebooks.decision.model import DecisionModel

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT / "configs/decision/tasks/triage.json").read_text())


def test_training_and_cli_without_external_decision_or_retrieval_packages(tmp_path):
    script = """
import importlib.abc
import json
import sys
from pathlib import Path
import torch
import yaml
root, folder = map(Path, sys.argv[1:])
blocked = ("laya", "advai_notebooks.embedding", "advai_notebooks.ranker", "sentence_transformers", "advai_notebooks.decision.providers")
class BlockDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise AssertionError("Forbidden dependency: " + fullname)
sys.meta_path.insert(0, BlockDependencies())
from advai_notebooks.decision.pipeline import run_experiment
from advai_notebooks.decision.model import DecisionModel
from advai_notebooks.decision.inference import predict_state
from advai_notebooks.decision.cli import main
raw = yaml.safe_load((root / "configs/decision/smoke.yaml").read_text())
raw["output_dir"] = str(folder / "run")
config = folder / "config.yaml"
config.write_text(yaml.safe_dump(raw))
report = run_experiment(config, root=root)
resumed = run_experiment(config, root=root, resume=folder / "run/latest")
assert resumed["training"] == report["training"]
bundle = DecisionModel.load(report["export_dir"])
tasks = json.loads((Path(report["export_dir"]) / "tasks.json").read_text())
result = predict_state(bundle, "charged twice refund today", tasks, "triage", torch.device("cpu"))
assert {value["type"] for value in result["answers"].values()} == {"choice", "score", "noul"}
main(["validate", "--config", str(config), "--root", str(root)])
main(["evaluate", "--config", str(config), "--root", str(root), "--model", report["export_dir"]])
main(["predict", "--model", report["export_dir"], "--task", "triage", "--device", "cpu", "--state", "charged twice"])
main(["publish", "--export-dir", report["export_dir"], "--repo-id", "test/decision", "--dry-run"])
assert not any(any(name == prefix or name.startswith(prefix + ".") for prefix in blocked) for name in sys.modules)
print("NATIVE_INDEPENDENCE_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_INDEPENDENCE_OK" in result.stdout


def test_option_order_and_batch_padding_do_not_change_answers():
    from advai_notebooks.decision.runtime import forward, make_batch

    bundle = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    bundle.model.eval()
    state = "charged twice refund today"
    actual = predict_state(bundle, state, TASKS, "triage", torch.device("cpu"))
    reordered = copy.deepcopy(TASKS)
    reordered["triage"]["department"]["criteria"] = dict(
        reversed(list(TASKS["triage"]["department"]["criteria"].items()))
    )
    result = predict_state(bundle, state, reordered, "triage", torch.device("cpu"))
    assert result["answers"]["department"]["probabilities"] == pytest.approx(
        actual["answers"]["department"]["probabilities"], abs=1e-7
    )
    rows, tasks = load_data(
        ROOT / "data/sample/decision/triage.jsonl", ROOT / "configs/decision/tasks/triage.json"
    )
    items = prepare_items(rows[:1], tasks, bundle.tokenizer, 128, 96)
    batch = make_batch(items, bundle.tokenizer.pad_token_id, torch.device("cpu"))
    logits = forward(bundle.model, batch)
    assert logits.shape == (3, 3) and logits[-1, -1] == -1e4
    batch["input_ids"][~batch["option_mask"]] = 999999
    assert torch.equal(logits, forward(bundle.model, batch))


def test_native_proper_rewards_prefer_target_distribution_and_ordered_errors():
    target = torch.tensor([[0.25, 0.75, 0.0]])
    mask = torch.tensor([[True, True, True]])
    for kind in (0, 1, 2):
        optimum = proper_reward(target, target, torch.tensor([kind]), mask)
        wrong = proper_reward(torch.tensor([[1.0, 0.0, 0.0]]), target, torch.tensor([kind]), mask)
        assert optimum > wrong
    target = torch.tensor([[1.0, 0.0, 0.0]])
    near, far = torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0.0, 0.0, 1.0]])
    assert proper_reward(near, target, torch.tensor([1]), mask) > proper_reward(
        far, target, torch.tensor([1]), mask
    )


def test_temperature_fit_improves_calibration_nll_and_sparse_types_fall_back():
    from advai_notebooks.decision.config import CalibrationConfig

    rows = [
        {"qtype": 0, "logits": [5.0, 0.0], "target": [1.0, 0.0] if i < 7 else [0.0, 1.0]}
        for i in range(10)
    ]
    result = fit_calibration(rows, CalibrationConfig(min_bucket_samples=5))
    temp = result["temperature_by_options"]["0:2"]
    logits = torch.tensor([row["logits"] for row in rows])
    targets = torch.tensor([row["target"] for row in rows])
    initial = -(targets * logits.log_softmax(-1)).sum(-1).mean()
    fitted = -(targets * (logits / temp).log_softmax(-1)).sum(-1).mean()
    assert fitted < initial and temp > 1
    assert result["temperature"][1:] == [1.0, 1.0] and result["unfitted_types"] == [1, 2]
    assert fit_calibration(rows[:2])["temperature"] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("architecture,pooling", [("bert", "mean"), ("roberta", "cls")])
def test_local_hf_encoder_trains_head_while_freezing_backbone(tmp_path, architecture, pooling):
    from advai_notebooks.decision.pipeline import run_experiment

    source = DecisionModel.from_config(load_config(ROOT / "configs/decision/smoke.yaml"), TASKS)
    tokenizer, encoder = source.tokenizer, source.model.encoder
    if architecture == "roberta":
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

        words = [word for word in tokenizer.get_vocab() if not word.startswith("[")]
        vocab = {word: i for i, word in enumerate(["<s>", "<pad>", "</s>", "[UNK]", *words])}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[("<s>", 0), ("</s>", 2)],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            pad_token="<pad>",
            model_input_names=["input_ids", "attention_mask"],
        )
        encoder = RobertaModel(
            RobertaConfig(
                vocab_size=len(tokenizer),
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=130,
                pad_token_id=1,
            )
        )
    base = tmp_path / "encoder"
    encoder.save_pretrained(base)
    tokenizer.save_pretrained(base)
    raw = yaml.safe_load((ROOT / "configs/decision/smoke.yaml").read_text())
    raw["model"].update({"id": str(base), "freeze_encoder": True, "pooling": pooling})
    raw["data"] = {
        key: str(ROOT / value) if key in {"path", "tasks"} else value
        for key, value in raw["data"].items()
    }
    raw["output_dir"] = str(tmp_path / "run")
    raw["training"].update({"loss": "ce", "gradient_checkpointing": True})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    report = run_experiment(path)
    latest = DecisionModel.load(tmp_path / "run/latest")
    before = DecisionModel.load(tmp_path / "run/best") if report["best_epoch"] == 0 else None
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, latest.model.encoder.state_dict()[key]), key
    if before is not None:
        assert any(
            not torch.equal(value, latest.model.head.state_dict()[key])
            for key, value in before.model.head.state_dict().items()
        )
    assert latest.source["model_id"] == str(base)
