"""Independent task metrics with consistent report and selection behavior."""

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from advai_notebooks.decision import metrics as decision_module
from advai_notebooks.decision.metrics import evaluate_predictions
from advai_notebooks.embedding import metrics as embedding_module
from advai_notebooks.ranker import metrics as ranker_module


@pytest.mark.parametrize("task", ["embedding", "ranker", "decision"])
def test_metrics_work_without_common_or_other_task_metrics(task):
    code = """
import importlib
import importlib.abc
import sys

task = sys.argv[1]
blocked = {"advai_notebooks.common.metrics", "advai_notebooks.common.ranking_metrics"}
blocked.update("advai_notebooks." + other for other in ("embedding", "ranker", "decision") if other != task)
class BlockMetrics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Forbidden metric dependency: " + fullname)
sys.meta_path.insert(0, BlockMetrics())
module = importlib.import_module("advai_notebooks." + task + ".metrics")
record = module.metric_record("quality", 0.5, 2, {"scope": "overall"}, "minimize")
report = {"metric_schema_version": 1, "metric_records": [record]}
assert module.select_metric(report, "quality") == record
assert module.is_improvement(0.4, 0.5, record["direction"])
assert not module.is_improvement(0.5, 0.5, record["direction"])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, task],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def ranking_data():
    return SimpleNamespace(
        corpus={key: {} for key in ("a", "b", "z")},
        qrels={"q1": {"a": 3, "b": 1, "z": 0}, "q2": {"a": 2, "z": 0}},
        query_ids=lambda split: ["q1", "q2"],
        positives=lambda qid: {"a", "b"} if qid == "q1" else {"a"},
    )


@pytest.mark.parametrize(
    "task,module", [("embedding", embedding_module), ("ranker", ranker_module)]
)
def test_task_ranking_records_preserve_full_qrels_denominator_and_values(task, module):
    data = ranking_data()
    rankings = {
        "q1": [{"doc_id": "z", "score": 2}, {"doc_id": "b", "score": 1}],
        "q2": [{"doc_id": "a", "score": 3}],
    }
    report = module.evaluate_rankings(rankings, data, "test", [1, 2])
    assert report["n_queries"] == 2
    assert report["unjudged_policy"] == "zero_gain_for_offline_metrics"
    assert report["metrics"]["mrr@2"] == 0.75
    assert report["metrics"]["ndcg@1"] == 0.5
    assert report["task"] == task
    selected = module.select_metric(report, "recall@2")
    assert selected == {
        "name": "recall@2",
        "value": 0.75,
        "n": 2,
        "group": {"scope": "overall"},
        "direction": "maximize",
    }
    # The absent positive a stays in q1's recall denominator/ideal ranking.
    assert report["per_query"]["q1"]["recall@2"] == 0.5
    assert report["per_query"]["q1"]["ndcg@2"] == pytest.approx(
        (1 / math.log2(3)) / (7 + 1 / math.log2(3))
    )
    per_query = [
        record
        for record in report["metric_records"]
        if record["group"] == {"scope": "query", "id": "q1"}
    ]
    assert len(per_query) == 6 and all(record["n"] == 1 for record in per_query)
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_decision_metric_counts_use_actual_ordinal_population_and_grouping():
    rows = [
        {
            "task": "triage",
            "qid": "department",
            "qtype": 0,
            "probabilities": [0.8, 0.2],
            "target": [1.0, 0.0],
        },
        {
            "task": "triage",
            "qid": "urgency",
            "qtype": 1,
            "probabilities": [0.1, 0.2, 0.7],
            "target": [1.0, 0.0, 0.0],
        },
        {
            "task": "triage",
            "qid": "urgency",
            "qtype": 1,
            "probabilities": [0.2, 0.5, 0.3],
            "target": [0.0, 1.0, 0.0],
        },
        {
            "task": "triage",
            "qid": "churn",
            "qtype": 2,
            "probabilities": [0.4, 0.6],
            "target": [0.0, 1.0],
        },
    ]
    report = evaluate_predictions(rows)
    score = decision_module.select_metric(report, "normalized_score_mae")
    assert score["n"] == 2 and score["value"] == pytest.approx(0.425)
    assert score["direction"] == "minimize"
    nll = decision_module.select_metric(report, "nll")
    assert nll["n"] == 4 and nll["value"] == pytest.approx(
        -sum(map(math.log, [0.8, 0.1, 0.5, 0.6])) / 4
    )
    assert decision_module.select_metric(report, "accuracy")["direction"] == "maximize"
    score_groups = [
        record for record in report["metric_records"] if record["name"] == "normalized_score_mae"
    ]
    assert {record["group"]["scope"] for record in score_groups} == {"overall", "question", "type"}
    assert all(record["n"] == 2 for record in score_groups)
    assert not any(
        record["name"] == "normalized_score_mae" and record["group"].get("id") == "choice"
        for record in report["metric_records"]
    )
    assert report["overall"]["normalized_score_mae"] == score["value"]
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("module", [embedding_module, ranker_module, decision_module])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "schema", "nonfinite"])
def test_selection_requires_a_unique_valid_overall_record(module, mutation):
    overall = module.metric_record("quality", 0.4, 2, {"scope": "overall"}, "maximize")
    subgroup = module.metric_record(
        "quality", 0.9, 1, {"scope": "type", "id": "choice"}, "maximize"
    )
    report = {"metric_schema_version": 1, "metric_records": [overall, subgroup]}
    assert module.select_metric(report, "quality") == overall
    if mutation == "missing":
        report["metric_records"].remove(overall)
    elif mutation == "duplicate":
        report["metric_records"].append(overall)
    elif mutation == "schema":
        report["metric_schema_version"] = 2
    else:
        overall["value"] = float("nan")
    with pytest.raises(ValueError):
        module.select_metric(report, "quality")


@pytest.mark.parametrize("module", [embedding_module, ranker_module, decision_module])
@pytest.mark.parametrize(
    "setting,value",
    [("n", 0), ("n", True), ("value", float("inf")), ("direction", "automatic"), ("group", {})],
)
def test_metric_record_rejects_ambiguous_metadata(module, setting, value):
    fields = {
        "name": "quality",
        "value": 0.5,
        "n": 2,
        "group": {"scope": "overall"},
        "direction": "maximize",
    }
    fields[setting] = value
    with pytest.raises(ValueError):
        module.metric_record(**fields)


@pytest.mark.parametrize("module", [embedding_module, ranker_module])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_training_uses_task_callbacks_for_an_arbitrary_metric_name(tmp_path, module, direction):
    import torch

    from advai_notebooks.common.retrieval_training import train

    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)

    def save(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(path) / "weights.pt")

    adapter = SimpleNamespace(model=model, device=torch.device("cpu"), save=save)
    cfg = SimpleNamespace(
        kind="toy",
        runtime=SimpleNamespace(precision="fp32"),
        training=SimpleNamespace(
            epochs=2, gradient_checkpointing=False, gradient_accumulation=1, max_grad_norm=10.0
        ),
        evaluation=SimpleNamespace(selection_metric="quality"),
    )

    def evaluate():
        value = (
            float((model.weight.detach() - 1).square().item())
            if direction == "minimize"
            else float(model.weight.detach().item())
        )
        return {
            "metrics": {"quality": value},
            "metric_schema_version": 1,
            "metric_records": [
                module.metric_record("quality", value, 1, {"scope": "overall"}, direction)
            ],
        }

    result = train(
        adapter,
        cfg,
        lambda epoch: [[0]],
        lambda batch: (model.weight - 1).square().mean(),
        evaluate,
        tmp_path,
        optimizer_factory=lambda adapter, settings: torch.optim.SGD(model.parameters(), lr=0.1),
        metric_selector=module.select_metric,
        metric_comparator=module.is_improvement,
        signature="toy",
    )
    assert result["best_epoch"] == 2 and result["selection_direction"] == direction
    weights = torch.load(tmp_path / "best/weights.pt", weights_only=True)
    assert weights["weight"].item() == pytest.approx(0.36)
    state = torch.load(tmp_path / "latest/training_state.pt", weights_only=True)
    assert state["selection_direction"] == direction
