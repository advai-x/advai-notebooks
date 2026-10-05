"""Component composition preserves pooling semantics, learned tails and exports."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from advai_notebooks.embedding.config import load_config
from advai_notebooks.embedding.model import EmbeddingModel

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_components_compute_expected_normalized_pooling(pooling, tmp_path):
    from sentence_transformers import SentenceTransformer

    cfg = load_config(ROOT / "configs/embedding/smoke.yaml")
    settings = replace(cfg.model, pooling=pooling, query_prefix="query ")
    model = EmbeddingModel(settings, "cpu")
    model.model.eval()
    texts = ["reset password", "card charged twice"]
    features = model.features(texts, "query")
    inputs = {key: features[key] for key in ("input_ids", "attention_mask", "token_type_ids")}
    with torch.no_grad():
        tokens = model.backbone(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = tokens[:, 0] if pooling == "cls" else (tokens * mask).sum(1) / mask.sum(1)
        expected = torch.nn.functional.normalize(pooled, dim=1)
    actual = model.encode(texts, "query")
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
    assert model.backbone is model.model[0].auto_model
    assert all(view is module for view, module in zip(model.head.modules, tuple(model.model)[1:]))
    keys = list(model.model.state_dict())
    model.save(tmp_path / "export")
    restored = EmbeddingModel.load(tmp_path / "export")
    sdk = SentenceTransformer(str(tmp_path / "export"), device="cpu")
    assert list(restored.model.state_dict()) == keys == list(sdk.state_dict())
    torch.testing.assert_close(restored.encode(texts, "query"), actual, atol=0, rtol=0)


def test_head_retains_learned_dense_tail_and_its_gradients(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Dense, Normalize

    cfg = load_config(ROOT / "configs/embedding/smoke.yaml")
    original = EmbeddingModel(cfg.model, "cpu")
    dense = Dense(32, 16)
    sdk = SentenceTransformer(
        modules=[original.model[0], original.model[1], dense, Normalize()], device="cpu"
    )
    sdk.save_pretrained(str(tmp_path / "source"), create_model_card=False)
    model = EmbeddingModel(replace(cfg.model, id=str(tmp_path / "source")), "cpu")
    assert isinstance(model.head.modules[1], Dense)
    texts = ["payment refund", "reset password"]
    model.model.eval()
    expected = model.model(model.features(texts, "query"))["sentence_embedding"]
    torch.testing.assert_close(model.encode(texts, "query"), expected.detach(), atol=0, rtol=0)
    assert expected.shape == (2, 16)
    model.model.zero_grad()
    component_output = model.head(model._backbone_component(model.features(texts, "query")))[
        "sentence_embedding"
    ]
    component_output[:, 0].sum().backward()
    assert model.head.modules[1].linear.weight.grad.abs().sum() > 0
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    model.save(tmp_path / "export")
    restored = EmbeddingModel.load(tmp_path / "export")
    assert list(restored.model.state_dict()) == list(model.model.state_dict())
    torch.testing.assert_close(restored.encode(texts, "query"), expected.detach(), atol=0, rtol=0)
