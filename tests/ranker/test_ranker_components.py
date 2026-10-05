"""Native classification order, gradients and export paths survive component views."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from advai_notebooks.common.tiny import write_tiny_checkpoint
from advai_notebooks.ranker.config import load_config
from advai_notebooks.ranker.model import RankerModel

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("architecture", ["bert", "roberta"])
def test_native_classifier_order_dropout_gradients_and_exports(tmp_path, architecture):
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification

    source = write_tiny_checkpoint(tmp_path / "source", 128, classification=True)
    if architecture == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(source)
        classifier = RobertaForSequenceClassification(
            RobertaConfig(
                vocab_size=len(tokenizer),
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=130,
                num_labels=1,
                type_vocab_size=2,
                pad_token_id=tokenizer.pad_token_id,
                hidden_dropout_prob=0.2,
                attention_probs_dropout_prob=0.15,
                classifier_dropout=0.3,
            )
        )
        classifier.save_pretrained(source)
    cfg = load_config(ROOT / "configs/ranker/smoke.yaml")
    settings = replace(
        cfg.model, id=str(source), query_prefix="query ", document_prefix="document "
    )
    model = RankerModel(settings, "cpu")
    native = model.model.model
    assert model.backbone is native.base_model
    assert model.head.modules == {
        name: module for name, module in native.named_children() if module is not native.base_model
    }
    pairs = [("payment refund", "charged twice"), ("app crash", "reset password")]
    model.model.train()
    events = []
    modules = {"encoder": model.backbone, **model.head.modules}
    handles = [
        module.register_forward_hook(lambda module, args, output, name=name: events.append(name))
        for name, module in modules.items()
    ]
    try:
        rng = torch.get_rng_state()
        actual = model.logits(pairs)
        order = list(events)
        actual.square().sum().backward()
        gradients = {
            name: p.grad.clone() for name, p in model.model.named_parameters() if p.grad is not None
        }
        assert order[0] == "encoder" and "classifier" in order
        assert any(p.grad is not None for p in model.backbone.parameters())
        assert any(
            p.grad is not None
            for module in model.head.modules.values()
            for p in module.parameters()
        )
        model.model.zero_grad()
        events.clear()
        torch.set_rng_state(rng)
        expected = model.model(model.features(pairs))["scores"].reshape(-1)
        expected.square().sum().backward()
        assert events == order
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name, parameter in model.model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=0, rtol=0)
    finally:
        for handle in handles:
            handle.remove()
    keys = list(model.model.state_dict())
    scores = model.score(pairs)
    model.save(tmp_path / "export")
    restored = RankerModel.load(tmp_path / "export")
    sdk = CrossEncoder(str(tmp_path / "export"), device="cpu")
    assert list(restored.model.state_dict()) == keys == list(sdk.state_dict())
    assert restored.score(pairs) == scores
    torch.testing.assert_close(
        torch.tensor(scores),
        torch.tensor(
            sdk.predict(
                restored.prepare(pairs), show_progress_bar=False, activation_fn=torch.nn.Identity()
            ).reshape(-1)
        ),
        atol=1e-7,
        rtol=0,
    )
