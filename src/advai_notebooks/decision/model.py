"""Decision model facade: network composition, loading, features and typed inference."""

import inspect
import json
import re
import tempfile
from pathlib import Path

from torch import nn

from advai_notebooks.common.artifacts import write_json

from .backbones import build_backbone
from .heads import build_head


class DecisionModel(nn.Module):
    def __init__(
        self,
        encoder=None,
        pooling="mean",
        head_hidden_size=None,
        dropout=0.1,
        *,
        head_name="independent",
        backbone=None,
        head=None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else build_backbone(encoder, pooling)
        self.head = (
            head
            if head is not None
            else build_head(head_name, self.backbone.output_size, head_hidden_size, dropout)
        )

    @property
    def encoder(self):
        return self.backbone.encoder

    def forward(self, input_ids, attention_mask, option_mask, qtype, token_type_ids=None):
        features = self.backbone(input_ids, attention_mask, option_mask, token_type_ids)
        logits = self.head(features, option_mask, qtype)
        if logits.shape != option_mask.shape:
            raise ValueError("Decision head must return [batch, options] logits")
        return logits.float().masked_fill(~option_mask, -1e4)

    @property
    def model(self):
        """Expose the trainable module, as the embedding/ranker adapters do."""
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def _bind_metadata(self, tokenizer, config, source, tasks=None):
        from copy import deepcopy

        from .schema import validate_tasks

        # Runtime metadata is not a registered module/buffer and does not change
        # native parameter names.
        self.tokenizer, self.config, self.source = tokenizer, config, source
        self.task_definitions = deepcopy(validate_tasks(tasks)) if tasks is not None else None
        return self

    def _require_metadata(self):
        if not hasattr(self, "tokenizer"):
            raise ValueError(
                "Load a DecisionModel or use from_config before encoding/inference/export"
            )

    @classmethod
    def from_config(cls, cfg, tasks, device="cpu"):
        from transformers import AutoModel, AutoTokenizer

        settings = cfg.model
        if settings.id == "tiny":
            encoder, tokenizer = _tiny_encoder(settings.max_len, tasks)
            source = {"model_id": "tiny", "random_initialization": True, "provider": "hf"}
        else:
            local = cfg.path(settings.id)
            model_id = str(local) if local.is_dir() else settings.id
            kwargs = {"revision": settings.revision}
            if settings.subfolder:
                kwargs["subfolder"] = settings.subfolder
            tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            encoder = AutoModel.from_pretrained(model_id, **kwargs)
            source = {
                "model_id": model_id,
                "requested_revision": settings.revision,
                "resolved_revision": getattr(encoder.config, "_commit_hash", None),
                "provider": "hf",
                "head_initialization": "random",
            }
        _validate_encoder(encoder, tokenizer, settings.max_len)
        model = (
            cls(
                encoder,
                settings.pooling,
                settings.head_hidden_size,
                settings.dropout,
                head_name=settings.head,
            )
            .float()
            .to(device)
        )
        return model._bind_metadata(tokenizer, _model_config(settings), source, tasks)

    @classmethod
    def load(cls, path, device="cpu", *, weights=None):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        from .decoding import build_decoder
        from .encoding import build_input_encoder

        path = Path(path)
        config_path = path / "decision_config.json"
        if not config_path.is_file():
            raise ValueError(
                "Expected a native decision export with decision_config.json; retrain legacy checkpoints"
            )
        config = json.loads(config_path.read_text())
        version = config.get("schema_version")
        if version not in {1, 2} or config.get("architecture") != "hf_option_scorer":
            raise ValueError("Unsupported decision export schema or architecture")
        if version == 1:
            config.update({"encoding": "pair", "head": "independent", "decoder": "typed"})
        config["schema_version"] = 2
        build_decoder(config["decoder"])
        encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(path / "encoder", local_files_only=True)
        )
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
        _validate_encoder(encoder, tokenizer, config["max_len"])
        build_input_encoder(
            tokenizer, config["max_len"], config["head_max_len"], name=config["encoding"]
        )
        model = cls(
            encoder,
            config["pooling"],
            config["head_hidden_size"],
            config["dropout"],
            head_name=config["head"],
        )
        weights = weights if weights is not None else load_file(str(path / "model.safetensors"))
        if version == 1:
            weights = {_migrate_v1_key(key): value for key, value in weights.items()}
        model.load_state_dict(weights, strict=True)
        task_path = path / "tasks.json"
        tasks = json.loads(task_path.read_text()) if task_path.is_file() else None
        source = json.loads((path / "source.json").read_text())
        return model.float().to(device)._bind_metadata(tokenizer, config, source, tasks)

    def save(self, path):
        from safetensors.torch import save_file

        self._require_metadata()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        weights = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in self.state_dict().items()
        }
        temporary = path / "model.safetensors.tmp"
        save_file(weights, str(temporary))
        temporary.replace(path / "model.safetensors")
        self.encoder.config.save_pretrained(path / "encoder")
        self.tokenizer.save_pretrained(path / "tokenizer")
        write_json(path / "decision_config.json", self.config)
        write_json(path / "source.json", self.source)
        if self.task_definitions is not None:
            write_json(path / "tasks.json", self.task_definitions)
        return path

    def prepare(self, requests):
        from .data import prepare_requests

        return prepare_requests(requests)

    def features(self, requests, reject_truncation=True):
        from .encoding import build_input_encoder

        self._require_metadata()
        encode = build_input_encoder(
            self.tokenizer,
            self.config["max_len"],
            self.config["head_max_len"],
            reject_truncation=reject_truncation,
            name=self.config.get("encoding", "pair"),
        )
        items = []
        for request in self.prepare(requests):
            encoded = encode(request)
            items.append(
                {"options": encoded.options, "qtype": request.qtype, "truncated": encoded.truncated}
            )
        return items

    def check_lengths(self, requests):
        self.features(requests, reject_truncation=True)

    def predict_items(self, items, batch_size=32, precision="fp32", calibrated=True):
        """Raw logits and probabilities; input items may additionally carry gold."""
        import torch

        from .calibrate import temp_bucket
        from .runtime import amp_context, forward, make_batch

        self._require_metadata()
        if not items:
            raise ValueError("Prediction set is empty")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.eval()
        predictions = []
        temperatures = self.config.get("temperature", [1.0] * 3)
        buckets = self.config.get("temperature_by_options", {})
        with torch.inference_mode():
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                batch = make_batch(chunk, self.tokenizer.pad_token_id, self.device)
                with amp_context(self.device, precision):
                    logits = forward(self, batch)
                for item, values in zip(chunk, logits):
                    k = len(item["options"])
                    values = values[:k].float().cpu()
                    temperature = (
                        buckets.get(temp_bucket(item["qtype"], k), temperatures[item["qtype"]])
                        if calibrated
                        else 1.0
                    )
                    probabilities = torch.softmax(values / temperature, -1).tolist()
                    predictions.append(
                        {
                            key: item[key]
                            for key in ("id", "qid", "task", "qtype", "target", "truncated")
                            if key in item
                        }
                        | {"logits": values.tolist(), "probabilities": probabilities}
                    )
        return predictions

    def system_one(self, inputs, task=None, *, tasks=None, batch_size=8, min_confidence=None):
        """Infer typed answers from requests, or a state and named task.

        system_one(requests) returns an answer list. system_one(state, task=...)
        returns {"answers": {question_id: answer}} using the model's task definitions,
        unless explicit tasks are supplied. Neither form requires labels.
        """
        from .data import requests_from_tasks

        self._require_metadata()
        if task is None:
            if tasks is not None:
                raise ValueError("task is required when supplying task definitions")
            return self._predict_requests(inputs, batch_size, min_confidence)
        definitions = self.task_definitions if tasks is None else tasks
        if definitions is None:
            raise ValueError(
                "This model has no task definitions; supply tasks or use DecisionRequest inputs"
            )
        requests = requests_from_tasks(inputs, definitions, task)
        answers = self._predict_requests(requests, batch_size, min_confidence)
        return {"answers": dict(zip(definitions[task], answers))}

    def _predict_requests(self, requests, batch_size=8, min_confidence=None):
        from .decoding import build_decoder, validate_confidence

        validate_confidence(min_confidence)
        requests = self.prepare(requests)
        items = self.features(requests)
        predictions = self.predict_items(items, batch_size=batch_size)
        decode = build_decoder(self.config.get("decoder", "typed"))
        return [
            decode(request, prediction["probabilities"], min_confidence)
            for request, prediction in zip(requests, predictions)
        ]

    def predict(self, requests, batch_size=8, min_confidence=None):
        """Compatibility alias for system_one(requests)."""
        return self.system_one(requests, batch_size=batch_size, min_confidence=min_confidence)

    def predict_requests(self, requests, batch_size=8, min_confidence=None):
        return self.system_one(requests, batch_size=batch_size, min_confidence=min_confidence)

    def predict_state(self, state, tasks, task, min_confidence=None, batch_size=8):
        return self.system_one(
            state, task, tasks=tasks, batch_size=batch_size, min_confidence=min_confidence
        )


def _model_config(settings):
    return {
        "schema_version": 2,
        "architecture": "hf_option_scorer",
        "provider": "hf",
        "encoding": settings.encoding,
        "head": settings.head,
        "decoder": settings.decoder,
        "pooling": settings.pooling,
        "head_hidden_size": settings.head_hidden_size,
        "dropout": settings.dropout,
        "max_len": settings.max_len,
        "head_max_len": settings.head_max_len,
        "temperature": [1.0] * 3,
    }


def _validate_encoder(encoder, tokenizer, max_len):
    if getattr(encoder.config, "is_decoder", False) or getattr(
        encoder.config, "is_encoder_decoder", False
    ):
        raise ValueError(
            "Decision requires a bidirectional text encoder, not a decoder/seq2seq model"
        )
    if not hasattr(encoder.config, "hidden_size") or tokenizer.pad_token_id is None:
        raise ValueError("Encoder must expose hidden_size and its tokenizer must have a pad token")
    maximum = getattr(encoder.config, "max_position_embeddings", None)
    # RoBERTa-style embeddings reserve indices through padding_idx, unlike BERT.
    embeddings = getattr(encoder, "embeddings", None)
    padding_idx = getattr(getattr(embeddings, "position_embeddings", None), "padding_idx", None)
    if maximum and padding_idx is not None:
        maximum -= padding_idx + 1
    if maximum and max_len > maximum:
        raise ValueError(f"max_len exceeds encoder position limit {maximum}")


def _migrate_v1_key(key):
    """Native v1 encoder/scorer parameters keep identical computation in v2."""
    for old, new in (
        ("encoder.", "backbone.encoder."),
        ("type_embedding.", "head.type_embedding."),
        ("head.", "head.scorer."),
    ):
        if key.startswith(old):
            return new + key[len(old) :]
    raise ValueError(f"Unknown native v1 model parameter: {key}")


def _tiny_encoder(max_len, tasks):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = set(re.findall(r"[a-z0-9_]+|[^\w\s]", json.dumps(tasks).lower()))
    words.update(
        "choice score noul type question option criteria level false true invoice payment refund bug outage account urgent low medium high yes no the a is has customer please cancel charged twice service works ticket".split()
    )
    words.update(str(i) for i in range(100))
    with tempfile.TemporaryDirectory() as directory:
        vocab = Path(directory) / "vocab.txt"
        vocab.write_text(
            "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(words)) + "\n"
        )
        key = (
            "vocab" if "vocab" in inspect.signature(BertTokenizerFast).parameters else "vocab_file"
        )
        tokenizer = BertTokenizerFast(**{key: str(vocab)}, do_lower_case=True)
    encoder = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=max_len,
            pad_token_id=tokenizer.pad_token_id,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    )
    return encoder, tokenizer
