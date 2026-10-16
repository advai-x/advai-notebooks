"""Instruct-owned Trainer assembly, validation selection and complete checkpoints."""

import copy
from functools import partial

from transformers import Trainer, TrainerCallback
from transformers.trainer_callback import PrinterCallback

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.llm.moe import GradientAudit, NativeCheckpointTrainer, coefficient, describe
from advai_notebooks.llm.runtime import trainer_arguments

from .data import collator
from .evaluate import evaluate
from .losses import loss
from .metrics import improves
from .optimizer import optimizer
from .sampling import training_rows


class SelectionCallback(TrainerCallback):
    def __init__(self, model, prepared, cfg, state):
        self.model, self.prepared, self.cfg, self.record = model, prepared, cfg, state

    def on_epoch_end(self, args, state, control, **kwargs):
        validation = evaluate(
            self.model, self.prepared, self.cfg, "validation", generate_metrics=False
        )
        epoch = int(round(state.epoch))
        self.record["history"].append(
            {"epoch": epoch, "step": state.global_step, "validation": validation}
        )
        if improves(validation, self.record["best_validation"]):
            self.record.update(
                best_epoch=epoch,
                best_validation=validation,
                best_checkpoint=f"trainer/checkpoint-{state.global_step}",
            )
        self.record["last_epoch"] = epoch

    def on_save(self, args, state, control, **kwargs):
        output = self.cfg.path(self.cfg.output_dir)
        checkpoint = output / "trainer" / f"checkpoint-{state.global_step}"
        # Add facade metadata and (for offline LoRA) the immutable bundled base.
        self.model.save(checkpoint)
        write_json(checkpoint / "advai_resume.json", self.record)
        write_json(output / "latest.json", {"checkpoint": str(checkpoint.relative_to(output))})


def train_model(
    model,
    prepared,
    cfg,
    signature,
    baseline_validation,
    resume_path=None,
    resume_state=None,
    callbacks=(),
):
    record = (
        copy.deepcopy(resume_state)
        if resume_state
        else {
            "schema_version": 1,
            "signature": signature,
            "best_epoch": 0,
            "best_validation": baseline_validation,
            "best_checkpoint": "baseline",
            "last_epoch": 0,
            "history": [],
        }
    )
    inventory = describe(model.model)
    audits = []
    loss_function = loss
    if inventory:
        record.setdefault("moe_model", inventory)
        audits = [GradientAudit(record)]
        loss_function = partial(loss, aux_loss_coef=coefficient(model.model))
    if record["last_epoch"] >= cfg.training.epochs:
        return record
    model.model.config.use_cache = False
    trainer_class = NativeCheckpointTrainer if inventory else Trainer
    trainer = trainer_class(
        model=model.model,
        args=trainer_arguments(cfg, model.device),
        train_dataset=training_rows(prepared),
        data_collator=collator(model.tokenizer),
        processing_class=model.tokenizer,
        compute_loss_func=loss_function,
        optimizers=(optimizer(model, cfg.training), None),
        callbacks=[*audits, SelectionCallback(model, prepared, cfg, record), *callbacks],
    )
    trainer.remove_callback(PrinterCallback)
    # PEFT wrappers also need causal (shifted) target counts for accumulation.
    trainer._loss_shifts_labels = True
    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    return record
