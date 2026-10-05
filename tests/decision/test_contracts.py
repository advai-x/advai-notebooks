import copy
import json
from pathlib import Path

import pytest

from advai_notebooks.decision.config import load_config
from advai_notebooks.decision.data import load_data
from advai_notebooks.decision.metrics import evaluate_predictions
from advai_notebooks.decision.schema import target_distribution, validate_tasks

ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = ROOT / "configs/decision/tasks/triage.json"
TASKS = json.loads(TASK_PATH.read_text())


def test_typed_hard_and_soft_labels():
    questions = TASKS["triage"]
    assert target_distribution(questions["department"], {"choice": "technical"}) == [0, 1, 0]
    assert target_distribution(questions["urgency"], {"score": 2}) == [0, 0, 1]
    assert target_distribution(questions["churn_risk"], {"noul": 0.2}) == [0.8, 0.2]
    assert target_distribution(
        questions["department"], {"probabilities": {"billing": 3, "other": 1}}
    ) == [0.75, 0, 0.25]


@pytest.mark.parametrize(
    "gold",
    [
        {"choice": "missing"},
        {"probabilities": {"missing": 1}},
        {"probabilities": {"billing": 0}},
        {"probabilities": {"billing": -1}},
        {"probabilities": {"billing": float("nan")}},
        {"probabilities": {"billing": 1e308, "technical": 1e308}},
        {"choice": "billing", "probabilities": {"billing": 1}},
    ],
)
def test_invalid_gold_is_rejected(gold):
    with pytest.raises(ValueError):
        target_distribution(TASKS["triage"]["department"], gold)


@pytest.mark.parametrize("label", [1.5, -1, 3, True])
def test_score_labels_are_ordered_integer_levels(label):
    with pytest.raises(ValueError):
        target_distribution(TASKS["triage"]["urgency"], {"score": label})


def test_source_group_and_identical_state_cannot_cross_splits(tmp_path):
    records, _ = load_data(ROOT / "data/sample/decision/triage.jsonl", TASK_PATH)
    for key in ("group_id", "state"):
        rows = copy.deepcopy([records[0], records[12]])
        rows[1][key] = rows[0][key]
        path = tmp_path / "leaked.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with pytest.raises(ValueError, match="multiple splits"):
            load_data(path, TASK_PATH)


def test_unknown_task_and_duplicate_id_are_rejected(tmp_path):
    records, _ = load_data(ROOT / "data/sample/decision/triage.jsonl", TASK_PATH)
    for rows in ([records[0], records[0]], [{**records[0], "task": "missing"}]):
        path = tmp_path / "invalid.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with pytest.raises(ValueError):
            load_data(path, TASK_PATH)


def test_unknown_config_keys_do_not_silently_change_training(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  batch_szie: 2\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_task_schema_rejects_unknown_type():
    tasks = copy.deepcopy(TASKS)
    tasks["triage"]["department"]["type"] = "arbitrary"
    with pytest.raises(ValueError):
        validate_tasks(tasks)


def test_metrics_have_known_values_and_include_confidence_one():
    row = {
        "task": "triage",
        "qid": "urgency",
        "qtype": 1,
        "target": [0, 0, 1],
        "probabilities": [1, 0, 0],
    }
    result = evaluate_predictions([row])["overall"]
    assert result["accuracy"] == 0
    assert result["brier"] == 2
    assert result["ece"] == 1
    assert result["normalized_score_mae"] == 1
