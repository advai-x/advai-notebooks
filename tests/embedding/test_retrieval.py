import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from advai_notebooks.common.retrieval_config import load_config
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.data_preparation.mining import mine_negatives
from advai_notebooks.embedding.metrics import evaluate_rankings
from advai_notebooks.embedding.sampling import embedding_batches

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs/embedding/smoke.yaml", "embedding")


@pytest.fixture
def data(cfg):
    return load_data(cfg)


def test_metrics_use_all_positives_and_graded_ideal(data):
    ranks = {
        q: [
            {"doc_id": "profile-help", "score": 2.0},
            {"doc_id": "payment-guide", "score": 1.0},
            {"doc_id": "profile-guide", "score": 0.0},
        ]
        for q in data.query_ids("test")
    }
    result = evaluate_rankings(ranks, data, "test", [1, 3])["metrics"]
    assert result["recall@1"] == 0.5
    assert result["mrr@1"] == 1.0
    assert result["ndcg@1"] == pytest.approx(1 / 7)
    assert result["recall@3"] == 1.0
    assert result["ndcg@3"] == pytest.approx((1 + 7 / math.log2(4)) / (7 + 1 / math.log2(3)))


def test_missing_positive_is_not_injected_or_dropped(data):
    ranks = {q: [{"doc_id": "payment-guide", "score": 1.0}] for q in data.query_ids("test")}
    assert set(evaluate_rankings(ranks, data, "test", [1])["metrics"].values()) == {0.0}


def test_rejects_duplicate_ranked_document(data):
    ranks = {q: [{"doc_id": "profile-guide", "score": 1.0}] * 2 for q in data.query_ids("test")}
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_rankings(ranks, data, "test", [1])


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("query_group", "crosses splits"),
        ("query_text", "crosses splits"),
        ("doc_group", "source group crosses"),
        ("qrel", "Duplicate query/document"),
        ("positive", "positive judgment"),
        ("unknown", "unknown query/document"),
    ],
)
def test_rejects_bad_data(cfg, tmp_path, mutation, message):
    copied = {}
    for name in ("corpus", "queries", "qrels"):
        rows = [
            json.loads(line) for line in cfg.path(getattr(cfg.data, name)).read_text().splitlines()
        ]
        copied[name] = rows
    if mutation == "query_group":
        copied["queries"][-1]["group_id"] = copied["queries"][0]["group_id"]
    elif mutation == "query_text":
        copied["queries"][-1]["text"] = copied["queries"][0]["text"]
    elif mutation == "doc_group":
        copied["corpus"][-1]["group_id"] = copied["corpus"][0]["group_id"]
    elif mutation == "qrel":
        copied["qrels"].append(copied["qrels"][0])
    elif mutation == "positive":
        for row in copied["qrels"]:
            if row["query_id"] == "profile-0":
                row["relevance"] = 0
    else:
        copied["qrels"][0]["doc_id"] = "missing"
    for name, rows in copied.items():
        target = tmp_path / (name + ".jsonl")
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))
        setattr(cfg.data, name, str(target))
    with pytest.raises(ValueError, match=message):
        load_data(cfg)


def test_unknown_negatives_are_opt_in_and_heldout_docs_excluded(data):
    from advai_notebooks.data_preparation.config import load_config as load_preparation

    cfg = load_preparation(ROOT / "configs/data_preparation/smoke.yaml")
    cfg.mining.strategy = "embedding"

    class OrderedRetriever:
        def encode(self, texts, role, batch_size):
            import torch

            return torch.tensor([[1.0, 0.0]] * len(texts))

    for qid in data.query_ids("train"):
        positive = data.positives(qid)
        data.qrels[qid] = {
            doc: relevance
            for doc, relevance in data.qrels[qid].items()
            if doc in positive or doc.startswith("crash")
        }
    cfg.mining.negatives_per_query = 1
    # Crash queries have no judged negatives after this edit.
    with pytest.raises(ValueError, match="eligible negatives"):
        mine_negatives(OrderedRetriever(), data, cfg)
    cfg.data.allow_unjudged_negatives = True
    mined = mine_negatives(OrderedRetriever(), data, cfg)
    heldout = {d for d in data.corpus if d.startswith(("profile", "cancel"))}
    for row in mined:
        assert not set(row["negative_ids"]) & (data.positives(row["query_id"]) | heldout)


def test_batches_avoid_multi_positive_false_negatives(cfg, data):
    rows = [
        {"query_id": "payment-0", "positive_id": "payment-guide", "negative_ids": ["crash-guide"]},
        {"query_id": "invoice-0", "positive_id": "invoice-guide", "negative_ids": ["payment-help"]},
        {"query_id": "password-0", "positive_id": "password-guide", "negative_ids": ["crash-help"]},
    ]
    batches = embedding_batches(rows, data, 8, 7)
    for batch in batches:
        qids = {r["query_id"] for r in batch}
        assert not {"payment-0", "invoice-0"} <= qids
    assert sum(map(len, batches)) == len(rows)


def test_singleton_without_negatives_fails(data):
    with pytest.raises(ValueError, match="negative candidates"):
        embedding_batches(
            [{"query_id": "payment-0", "positive_id": "payment-guide", "negative_ids": []}],
            data,
            1,
            1,
        )


def test_chunked_retrieval_matches_full_and_ties_are_stable(cfg, data):
    import torch

    from advai_notebooks.common.retrieval import retrieve

    class Model:
        def encode(self, texts, role, batch_size):
            return torch.tensor([[1.0, float("payment" in text)] for text in texts])

    full = retrieve(Model(), data, data.query_ids("test"), 10, 100, 100)
    chunked = retrieve(Model(), data, data.query_ids("test"), 10, 3, 1)
    assert full == chunked


@pytest.mark.parametrize("kind", ["embedding", "ranker"])
def test_real_example_configs_parse(kind):
    cfg = load_config(ROOT / f"configs/{kind}/bge.yaml", kind)
    assert cfg.model.id.startswith("BAAI/")


def test_config_rejects_invalid_cutoff(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("evaluation:\n  cutoffs: [1, 0]\n")
    with pytest.raises(ValueError, match="positive integers"):
        load_config(config, "embedding")


@pytest.mark.parametrize("loss", ["mnrl", "cached_mnrl"])
def test_actual_embedding_training_and_native_export(cfg, tmp_path, loss):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from advai_notebooks.embedding.model import EmbeddingModel
    from advai_notebooks.embedding.pipeline import run

    cfg.output_dir = str(tmp_path / loss)
    cfg.training.epochs = 1
    cfg.training.loss = loss
    cfg.training.gradient_accumulation = 2
    cfg.training.gradient_checkpointing = True
    cfg.model.query_prefix, cfg.model.document_prefix = "query ", "document "
    report = run(cfg)
    assert len(report["training"]["history"]) == 2
    assert math.isfinite(report["training"]["history"][1]["train_loss"])
    before = EmbeddingModel.load(Path(cfg.output_dir) / "baseline")
    latest = EmbeddingModel.load(Path(cfg.output_dir) / "latest")
    assert any(
        not torch.equal(a, b) for a, b in zip(before.model.parameters(), latest.model.parameters())
    )
    adapter = EmbeddingModel.load(report["export"])
    sdk = SentenceTransformer(report["export"], device="cpu")
    for role in ("query", "document"):
        texts = ["reset password", "card charged twice"]
        expected = (
            sdk.encode_query(texts, show_progress_bar=False)
            if role == "query"
            else sdk.encode_document(texts, show_progress_bar=False)
        )
        np.testing.assert_allclose(adapter.encode(texts, role).numpy(), expected, atol=1e-6)
    with pytest.raises(ValueError, match="not empty"):
        run(cfg)


def test_lengths_fail_instead_of_silent_truncation(cfg):
    from advai_notebooks.embedding.model import EmbeddingModel

    model = EmbeddingModel(replace(cfg.model, max_len=8), "cpu")
    with pytest.raises(ValueError, match="exceeds max_len"):
        model.check_lengths(["payment " * 10], "query")
