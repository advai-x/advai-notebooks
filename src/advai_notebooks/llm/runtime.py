"""Neutral Trainer runtime, immutable run signatures and checkpoint verification."""

import importlib.metadata
import json
import platform
from pathlib import Path

from advai_notebooks.common.artifacts import digest
from advai_notebooks.common.provenance import source_fingerprint

PACKAGES = ("torch", "transformers", "peft", "accelerate", "tokenizers", "numpy", "PyYAML")


def environment(cfg, model):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
        "device": str(model.device),
        "precision": cfg.runtime.precision,
        "seed": cfg.training.seed,
        "source_sha256": source_fingerprint(cfg.kind),
        "model_source": model.source,
        "tiny_model": cfg.model.id in {"tiny", "tiny-olmoe", "tiny-qwen3-moe"},
    }


def signature(cfg, summary, model):
    return digest(
        {
            "config": cfg.to_dict(),
            "data": summary["fingerprint"],
            "environment": environment(cfg, model),
            "tokenizer": model.tokenizer.get_vocab(),
            "chat_template": model.tokenizer.chat_template,
        }
    )


def resume_checkpoint(output, resume):
    output = Path(output).resolve()
    path = Path(resume).expanduser().resolve()
    if path == output / "latest":
        path = output / json.loads((output / "latest.json").read_text())["checkpoint"]
    if not path.is_relative_to(output / "trainer") or not path.name.startswith("checkpoint-"):
        raise ValueError("Resume from a complete Trainer checkpoint of this experiment")
    required = (
        "advai_resume.json",
        "advai_metadata.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    if any(not (path / name).is_file() for name in required):
        raise ValueError("Checkpoint lacks complete model/optimizer/scheduler/RNG resume state")
    state = json.loads((path / "advai_resume.json").read_text())
    if state.get("schema_version") != 1:
        raise ValueError("Unknown resume schema")
    return path, state


def trainer_arguments(cfg, device):
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=str(cfg.path(cfg.output_dir) / "trainer"),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cpu=device.type == "cpu",
        fp16=cfg.runtime.precision == "fp16",
        bf16=cfg.runtime.precision == "bf16",
        seed=cfg.training.seed,
        data_seed=cfg.training.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=device.type == "cuda",
        remove_unused_columns=False,
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to=[],
        disable_tqdm=True,
        save_only_model=False,
    )
    if args.world_size != 1 or args.n_gpu > 1:
        raise ValueError("First-version LLM training supports a single device only")
    if args.device.type != device.type:
        raise ValueError(f"Trainer selected {args.device}, but runtime requested {device}")
    return args
