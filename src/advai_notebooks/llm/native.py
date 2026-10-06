"""HF/PEFT mechanics only; public facades, supervision and metrics belong to tasks."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.hub import resolve_export

from .configuration import GenerationSettings

TINY_TEMPLATE = """{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' }}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + '<|end|>\\n' }}{% endgeneration %}{% else %}{{ message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"""


def tiny_model():
    """An offline random decoder and fixed tokenizer, never a domain-quality model."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = (
        "the a is are of to and in for with you your I can please answer only "
        "help account password reset login email click settings support change "
        "cancel subscription profile name update billing payment card refund "
        "yes no true false one two three four 1 2 3 4 hello world blue red green "
        "sky water grass color what how where why when return JSON book read "
        "model text train data test validation document code language learn "
        "cat dog animal bird fish flies swims runs capital France Paris Italy "
        "Rome Germany Berlin sun hot ice cold instruction question response "
        "user assistant system explain compute add result 5 6 7 8 9 10 . , : ?"
    ).split()
    specials = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    ]
    vocab = {word: i for i, word in enumerate(dict.fromkeys([*specials, *words]))}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=specials[4:],
        model_max_length=128,
        chat_template=TINY_TEMPLATE,
    )
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    model.generation_config.eos_token_id = [3, 7]
    return model, tokenizer


def initialize(settings, training, root, device):
    local = Path(settings.id).expanduser()
    if not local.is_absolute():
        local = Path(root or ".") / local
    if settings.id == "tiny":
        model, tokenizer = tiny_model()
        source = {"id": "tiny", "revision": None}
    else:
        identifier = str(local.resolve()) if local.is_dir() else settings.id
        if local.is_dir() and (local / "adapter_config.json").is_file():
            raise ValueError("Training initialization requires a full or merged model")
        if local.is_dir() and (local / "advai_metadata.json").is_file():
            meta = json.loads((local / "advai_metadata.json").read_text())
            if meta.get("format") == "adapter":
                raise ValueError("Training initialization requires a full or merged model")
        config = AutoConfig.from_pretrained(
            identifier,
            revision=settings.revision,
            trust_remote_code=False,
        )
        if getattr(config, "quantization_config", None):
            raise ValueError("Quantized initial models are not supported in v1")
        model = AutoModelForCausalLM.from_pretrained(
            identifier,
            config=config,
            revision=settings.revision,
            trust_remote_code=False,
        )
        if getattr(model, "_hf_peft_config_loaded", False) or hasattr(model, "peft_config"):
            raise ValueError("Training initialization requires a full or merged model")
        tokenizer = AutoTokenizer.from_pretrained(
            identifier,
            revision=settings.revision,
            trust_remote_code=False,
            use_fast=True,
        )
        source = {
            "id": identifier,
            "revision": getattr(model.config, "_commit_hash", None) or settings.revision,
        }
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an EOS or padding token")
        tokenizer.pad_token = tokenizer.eos_token
    if settings.chat_template is not None:
        tokenizer.chat_template = settings.chat_template
    maximum = getattr(model.config, "max_position_embeddings", None)
    if maximum and settings.max_len > maximum:
        raise ValueError(f"max_len exceeds model context length {maximum}")
    temporary = None
    base = None
    if training and training.method == "lora":
        from peft import LoraConfig, get_peft_model

        if settings.id == "tiny" or local.is_dir():
            temporary = TemporaryDirectory(prefix="advai-lora-base-")
            model.save_pretrained(temporary.name, safe_serialization=True)
            base = {"kind": "bundled", "path": temporary.name}
        else:
            if not source["revision"] or len(source["revision"]) != 40:
                raise ValueError("LoRA requires a resolved immutable base model commit SHA")
            base = {"kind": "hub", **source}
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=training.lora_rank,
                lora_alpha=training.lora_alpha,
                lora_dropout=training.lora_dropout,
                target_modules=training.lora_target_modules,
                bias="none",
            ),
        )
        model.peft_config["default"].base_model_name_or_path = (
            base["path"] if base["kind"] == "bundled" else base["id"]
        )
        model.peft_config["default"].revision = base.get("revision")
    model.to(device)
    return model, tokenizer, source, base, temporary


def load_native(path, kind, device, *, trainable=False, **hub_options):
    root = resolve_export(path, **hub_options)
    meta = json.loads((root / "advai_metadata.json").read_text())
    if meta.get("schema_version") != 1 or meta.get("kind") != kind:
        raise ValueError(f"Expected a {kind} inference export")
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False, use_fast=True)
    base = meta.get("base_model")
    if meta["format"] == "adapter":
        from peft import PeftModel

        if base["kind"] == "bundled":
            base_path = root / base["path"]
            if not base_path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Bundled base escapes export directory")
            model = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=False)
            base = {"kind": "bundled", "path": str(base_path)}
        elif base["kind"] == "hub":
            model = AutoModelForCausalLM.from_pretrained(
                base["id"],
                revision=base["revision"],
                trust_remote_code=False,
                **{k: v for k, v in hub_options.items() if k != "revision"},
            )
        else:
            raise ValueError("Unknown adapter base format")
        model = PeftModel.from_pretrained(model, root, is_trainable=trainable)
        if (root / "generation_config.json").is_file():
            from transformers import GenerationConfig

            model.generation_config = GenerationConfig.from_pretrained(root)
        if base["kind"] == "bundled":
            model.peft_config["default"].base_model_name_or_path = base["path"]
    elif meta["format"] == "full":
        model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=False)
    else:
        raise ValueError("Unknown model export format")
    model.to(device)
    return model, tokenizer, meta, base


def save_native(facade, path, kind, *, merged=False):
    from peft import PeftModel

    root = Path(path).resolve()
    adapter = isinstance(facade.model, PeftModel)
    if merged and not adapter:
        raise ValueError("Only LoRA adapters can be exported as a merged model")
    expected = "adapter" if adapter and not merged else "full"
    existing = root / "advai_metadata.json"
    if existing.is_file() and json.loads(existing.read_text())["format"] != expected:
        raise ValueError("Choose a different directory for a different export format")
    if root.exists() and not existing.exists() and any(root.iterdir()):
        # Native Trainer checkpoints already contain model/optimizer state.
        if not root.name.startswith("checkpoint-") or not (root / "trainer_state.json").is_file():
            raise ValueError("Export directory must be empty or an existing same-format export")
    root.mkdir(parents=True, exist_ok=True)
    base = facade.base_source
    if merged:
        # Reload the adapter to avoid merge_and_unload mutating the live training model.
        with TemporaryDirectory(prefix="advai-merge-") as temporary:
            facade.save(temporary)
            model, _, _, _ = load_native(temporary, kind, "cpu")
            model = model.merge_and_unload(safe_merge=True)
            model.save_pretrained(root, safe_serialization=True)
        base = None
    else:
        facade.model.save_pretrained(root, safe_serialization=True)
        if adapter and base["kind"] == "bundled":
            destination = root / "base_model"
            if Path(base["path"]).resolve() != destination:
                shutil.copytree(base["path"], destination, dirs_exist_ok=True)
            base = {"kind": "bundled", "path": "base_model"}
        if adapter:
            # Keep PEFT's own config portable as well as our explicit loading contract.
            adapter_config = root / "adapter_config.json"
            config = json.loads(adapter_config.read_text())
            config["base_model_name_or_path"] = (
                "base_model" if base["kind"] == "bundled" else base["id"]
            )
            config["revision"] = base.get("revision")
            write_json(adapter_config, config)
    facade.tokenizer.save_pretrained(root)
    facade.model.generation_config.save_pretrained(root)
    write_json(
        root / "advai_metadata.json",
        {
            "schema_version": 1,
            "kind": kind,
            "format": expected,
            "model": asdict(facade.settings),
            "source": facade.source,
            "base_model": base,
            "generation": asdict(facade.generation),
        },
    )
    return root


def generate(facade, prompts, *, max_new_tokens=None, **overrides):
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(text, str) or not text.strip() for text in prompts)
    ):
        raise ValueError("Generation expects a nonempty list of nonempty prompts")
    allowed = {"do_sample", "temperature", "top_p"}
    if set(overrides) - allowed:
        raise ValueError(f"Unsupported generation settings: {set(overrides) - allowed}")
    settings = GenerationSettings(
        **{
            **asdict(facade.generation),
            **overrides,
            **({"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}),
        }
    )
    from advai_notebooks.common.configuration import positive_int, positive_number

    positive_int("max_new_tokens", settings.max_new_tokens)
    if type(settings.do_sample) is not bool:
        raise ValueError("do_sample must be boolean")
    positive_number("temperature", settings.temperature)
    positive_number("top_p", settings.top_p)
    if settings.top_p > 1:
        raise ValueError("top_p must be <= 1")
    old_padding = facade.tokenizer.padding_side
    facade.tokenizer.padding_side = "left"
    try:
        inputs = facade.tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
    finally:
        facade.tokenizer.padding_side = old_padding
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length + settings.max_new_tokens > facade.settings.max_len:
        raise ValueError("Prompt plus max_new_tokens exceeds configured max_len; no truncation")
    inputs = {
        k: v.to(facade.device) for k, v in inputs.items() if k in {"input_ids", "attention_mask"}
    }
    kwargs = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
        "pad_token_id": facade.tokenizer.pad_token_id,
        "eos_token_id": facade.model.generation_config.eos_token_id
        or facade.tokenizer.eos_token_id,
    }
    if settings.do_sample:
        kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
    training = facade.model.training
    cache = facade.model.config.use_cache
    facade.model.eval()
    facade.model.config.use_cache = True
    try:
        with torch.inference_mode():
            generated = facade.model.generate(**inputs, **kwargs)
    finally:
        facade.model.train(training)
        facade.model.config.use_cache = cache
    return facade.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)


def collate(rows, pad_token_id):
    """Pad by position, so real EOS labels survive when EOS is also padding."""
    width = max(len(row["input_ids"]) for row in rows)
    return {
        key: torch.tensor(
            [row[key] + [padding] * (width - len(row[key])) for row in rows], dtype=torch.long
        )
        for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))
    }
