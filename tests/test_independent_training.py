"""Behavioral proof that either task can run with the other task unavailable."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["embedding", "ranker"])
def test_training_export_inference_and_cli_without_other_task(tmp_path, kind):
    code = """
import importlib.abc
import importlib
import json
import sys
from pathlib import Path

root, output, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
other = "ranker" if kind == "embedding" else "embedding"
blocked = ("advai_notebooks." + other, "advai_notebooks.decision",
           "advai_notebooks.common.metrics", "advai_notebooks.common.ranking_metrics")
class BlockOther(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Forbidden cross-task import: " + fullname)
sys.meta_path.insert(0, BlockOther())
prefix = "advai_notebooks." + kind
config = importlib.import_module(prefix + ".config")
pipeline = importlib.import_module(prefix + ".pipeline")
cfg = config.load_config(root / "configs" / kind / "smoke.yaml")
cfg.output_dir, cfg.training.epochs = str(output / kind), 1
report = pipeline.run(cfg)
resumed = pipeline.run(cfg, resume=Path(report["export"]).parent / "latest")
assert resumed["training"] == report["training"]
models = importlib.import_module(prefix + ".model")
if kind == "embedding":
    model = models.EmbeddingModel.load(report["export"])
    assert model.encode(["reset password"], "query").shape == (1, 32)
else:
    model = models.RankerModel.load(report["export"])
    assert len(model.score([("reset password", "account login password reset")])) == 1
cli = importlib.import_module(prefix + ".cli")
cli.main(["validate", "--config", str(root / "configs" / kind / "smoke.yaml")])
cli.main(["publish", "--export-dir", report["export"], "--repo-id", "test/" + kind,
          "--dry-run"])
assert not any(n == "advai_notebooks." + other or n.startswith("advai_notebooks." + other + ".") for n in sys.modules)
print("INDEPENDENCE_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(tmp_path), kind],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INDEPENDENCE_OK" in result.stdout


@pytest.mark.parametrize(
    "strategy,candidates", [("judged", "all_corpus"), ("embedding", "embedding")]
)
def test_preparation_artifacts_feed_both_independent_loaders(tmp_path, strategy, candidates):
    from advai_notebooks.common.retrieval_data import load_data
    from advai_notebooks.data_preparation.config import load_config
    from advai_notebooks.data_preparation.pipeline import prepare
    from advai_notebooks.embedding.config import load_config as embedding_config
    from advai_notebooks.embedding.data import load_training_samples as embedding_samples
    from advai_notebooks.ranker.config import load_config as ranker_config
    from advai_notebooks.ranker.data import load_evaluation_candidates
    from advai_notebooks.ranker.data import load_training_samples as ranker_samples

    cfg = load_config(ROOT / "configs/data_preparation/smoke.yaml")
    cfg.output_dir = str(tmp_path / strategy)
    cfg.mining.strategy, cfg.mining.candidate_strategy = strategy, candidates
    result = prepare(cfg)
    emb, rank = (
        embedding_config(ROOT / "configs/embedding/smoke.yaml"),
        ranker_config(ROOT / "configs/ranker/smoke.yaml"),
    )
    emb.data.training_samples = result["embedding_samples"]
    rank.data.training_samples, rank.data.candidates = (
        result["ranker_samples"],
        result["candidates"],
    )
    data = load_data(emb)
    assert len(embedding_samples(emb, data)[0]) == 24
    assert {r["label"] for r in ranker_samples(rank, data)[0]} == {0.0, 1.0}
    loaded = load_evaluation_candidates(rank, data)
    assert set(loaded) == {"validation", "test"}
    heldout = {d for d in data.corpus if d.startswith(("cancel", "profile"))}
    assert not any(set(r["negative_ids"]) & heldout for r in embedding_samples(emb, data)[0])


@pytest.mark.parametrize("kind", ["embedding", "ranker"])
@pytest.mark.parametrize(
    "mutation", ["fingerprint", "heldout_query", "known_positive", "duplicate"]
)
def test_prepared_training_contracts_reject_leakage(tmp_path, kind, mutation):
    import importlib

    from advai_notebooks.common.retrieval_data import load_data

    config = importlib.import_module(f"advai_notebooks.{kind}.config").load_config(
        ROOT / f"configs/{kind}/smoke.yaml"
    )
    data = load_data(config)
    artifact = json.loads(config.path(config.data.training_samples).read_text())
    if mutation == "fingerprint":
        artifact["data_fingerprint"] = "different"
    elif mutation == "heldout_query":
        artifact["rows"][0]["query_id"] = "profile-0"
    elif mutation == "duplicate":
        artifact["rows"].append(artifact["rows"][0])
    elif kind == "embedding":
        artifact["rows"][0]["negative_ids"][0] = artifact["rows"][0]["positive_id"]
    else:
        positive = next(row for row in artifact["rows"] if row["label"] == 1)
        positive["label"] = 0
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(artifact))
    config.data.training_samples = str(path)
    with pytest.raises(ValueError):
        importlib.import_module(f"advai_notebooks.{kind}.data").load_training_samples(config, data)


@pytest.mark.parametrize(
    "mutation", ["fingerprint", "missing_query", "duplicate", "unknown_doc", "nonfinite"]
)
def test_external_candidates_checked(tmp_path, mutation):
    from advai_notebooks.common.retrieval_data import load_data
    from advai_notebooks.ranker.config import load_config
    from advai_notebooks.ranker.data import load_evaluation_candidates

    cfg = load_config(ROOT / "configs/ranker/smoke.yaml")
    data = load_data(cfg)
    artifact = json.loads(cfg.path(cfg.data.candidates).read_text())
    rows = artifact["splits"]["test"]["profile-0"]
    if mutation == "fingerprint":
        artifact["data_fingerprint"] = "different"
    elif mutation == "missing_query":
        del artifact["splits"]["test"]["profile-0"]
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "unknown_doc":
        rows[0]["doc_id"] = "missing"
    else:
        rows[0]["score"] = float("nan")
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(artifact))
    cfg.data.candidates = str(path)
    with pytest.raises(ValueError):
        load_evaluation_candidates(cfg, data)


def test_ranker_rejects_embedding_and_retriever_settings(tmp_path):
    from advai_notebooks.ranker.config import load_config

    for value in (
        "retrieval:\n  model: tiny\n",
        "training:\n  loss: cached_mnrl\n",
        "training:\n  scale: 20\n",
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(value)
        with pytest.raises(ValueError):
            load_config(path)
