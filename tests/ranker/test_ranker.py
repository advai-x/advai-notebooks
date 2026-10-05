import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from advai_notebooks.common.retrieval_config import load_config
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.common.runtime import setup
from advai_notebooks.ranker.model import RankerModel
from advai_notebooks.ranker.pipeline import run

ROOT = Path(__file__).resolve().parents[2]


def test_ranker_training_preserves_candidates_and_native_scores(tmp_path):
    import torch
    from sentence_transformers import CrossEncoder

    cfg = load_config(ROOT / "configs/ranker/smoke.yaml", "ranker")
    cfg.output_dir = str(tmp_path / "ranker")
    cfg.training.epochs = 1
    cfg.training.gradient_checkpointing = True
    cfg.model.query_prefix, cfg.model.document_prefix = "query ", "document "
    report = run(cfg)
    output = Path(cfg.output_dir)
    assert math.isfinite(report["training"]["history"][1]["train_loss"])
    baseline, latest = RankerModel.load(output / "baseline"), RankerModel.load(output / "latest")
    assert any(
        not torch.equal(a, b)
        for a, b in zip(baseline.model.parameters(), latest.model.parameters())
    )
    candidates = json.loads((output / "candidates.json").read_text())["splits"]["test"]
    rankings = json.loads((output / "test_rankings.json").read_text())
    for qid, rows in candidates.items():
        ids = {r["doc_id"] for r in rows}
        assert ids == {r["doc_id"] for r in rankings["baseline"][qid]}
        assert ids == {r["doc_id"] for r in rankings["finetuned"][qid]}
    model = RankerModel.load(report["export"])
    sdk = CrossEncoder(report["export"], device="cpu")
    pairs = [("payment refund", "card charged twice"), ("app crash", "reset password")]
    prepared = model.prepare(pairs)
    np.testing.assert_allclose(
        model.score(pairs),
        sdk.predict(prepared, show_progress_bar=False, activation_fn=torch.nn.Identity()).reshape(
            -1
        ),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="not empty"):
        run(cfg)


def test_evaluation_does_not_inject_a_positive():
    from advai_notebooks.ranker.pipeline import evaluate

    cfg = load_config(ROOT / "configs/ranker/smoke.yaml", "ranker")
    data = load_data(cfg)
    model = RankerModel(cfg.model, setup(cfg))
    candidates = {q: [{"doc_id": "payment-guide", "score": 1.0}] for q in data.query_ids("test")}
    report, rankings = evaluate(model, data, cfg, "test", candidates)
    assert set(report["metrics"].values()) == {0.0}
    assert all(len(rows) == 1 for rows in rankings.values())


def test_pair_truncation_checked():
    cfg = load_config(ROOT / "configs/ranker/smoke.yaml", "ranker")
    model = RankerModel(replace(cfg.model, max_len=8), "cpu")
    with pytest.raises(ValueError, match="exceeds max_len"):
        model.check_lengths([("payment " * 3, "refund " * 3)])


def test_both_retrievers_both_rankers_compare(tmp_path):
    from advai_notebooks.embedding.pipeline import run as run_embedding
    from advai_notebooks.workflows.config import load_config as load_workflow
    from advai_notebooks.workflows.retrieval import compare

    embedding = load_config(ROOT / "configs/embedding/smoke.yaml", "embedding")
    embedding.training.epochs = 1
    embedding.output_dir = str(tmp_path / "embedding")
    run_embedding(embedding)
    ranker = load_config(ROOT / "configs/ranker/smoke.yaml", "ranker")
    ranker.training.epochs = 1
    ranker.output_dir = str(tmp_path / "ranker")
    run(ranker)
    result = compare(
        load_workflow(ROOT / "configs/workflows/smoke.yaml"),
        embedding.output_dir,
        ranker.output_dir,
        tmp_path / "compare.json",
    )
    assert len(result["rows"]) == 6
    assert {(r["embedding"], r["ranker"]) for r in result["rows"]} == {
        (e, r) for e in ("baseline", "finetuned") for r in ("none", "baseline", "finetuned")
    }
