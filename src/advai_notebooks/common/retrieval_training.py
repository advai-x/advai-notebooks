"""Single-device training with task-supplied selection and checkpoint exports."""

import math
from pathlib import Path

from .artifacts import write_json
from .checkpoint import capture_rng, model_weights, restore_rng, save_training_state
from .runtime import amp_context


def train(
    adapter,
    cfg,
    batches_for_epoch,
    loss_for_batch,
    evaluate,
    output,
    *,
    optimizer_factory,
    metric_selector,
    metric_comparator,
    signature,
    resume_state=None,
):
    import torch

    output = Path(output)
    model = adapter.model
    if cfg.training.gradient_checkpointing:
        base = adapter.backbone
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = optimizer_factory(adapter, cfg.training)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.runtime.precision == "fp16")
    metric = cfg.evaluation.selection_metric
    if resume_state is None:
        baseline = evaluate()
        selected = metric_selector(baseline, metric)
        best, direction = selected["value"], selected["direction"]
        adapter.save(output / "baseline")
        adapter.save(output / "best")
        history = [{"epoch": 0, "train_loss": None, "validation": baseline["metrics"]}]
        best_epoch, first_epoch = 0, 1
        best_weights = model_weights(model)
    else:
        state = resume_state
        if state["signature"] != signature:
            raise ValueError("Resume configuration/data/implementation differ")
        direction = state.get("selection_direction")
        if direction not in {"maximize", "minimize"}:
            raise ValueError("Resume state requires an explicit selection_direction")
        model.load_state_dict(state["model_weights"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best, best_epoch = state["best_validation_score"], state["best_epoch"]
        history, first_epoch = state["history"], state["epoch"] + 1
        best_weights = state["best_model_weights"]
        # Repair a best export that may have been interrupted in an uncommitted epoch.
        model.load_state_dict(best_weights)
        adapter.save(output / "best")
        model.load_state_dict(state["model_weights"])
        restore_rng(state["rng"], adapter.device)
    for epoch in range(first_epoch, cfg.training.epochs + 1):
        batches = batches_for_epoch(epoch)
        if not batches:
            raise ValueError("No training batches")
        total_loss, total_examples = 0.0, 0
        model.train()
        # Accumulation weights microbatches by examples, including the last partial group.
        for group_start in range(0, len(batches), cfg.training.gradient_accumulation):
            group = batches[group_start : group_start + cfg.training.gradient_accumulation]
            examples = sum(len(batch) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                with amp_context(adapter.device, cfg.runtime.precision):
                    loss = loss_for_batch(batch)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                total_loss += loss.item() * len(batch)
                total_examples += len(batch)
                scaler.scale(loss * len(batch) / examples).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.max_grad_norm,
                error_if_nonfinite=not scaler.is_enabled(),
            )
            scaler.step(optimizer)
            scaler.update()
        validation = evaluate()
        mean_loss = total_loss / total_examples
        if not math.isfinite(mean_loss):
            raise ValueError("Non-finite epoch loss")
        history.append(
            {"epoch": epoch, "train_loss": mean_loss, "validation": validation["metrics"]}
        )
        selected = metric_selector(validation, metric)
        if selected["direction"] != direction:
            raise ValueError("Selection metric direction changed during training")
        if metric_comparator(selected["value"], best, direction):
            best, best_epoch = selected["value"], epoch
            best_weights = model_weights(model)
            adapter.save(output / "best")
        adapter.save(output / "latest")
        save_training_state(
            output / "latest" / "training_state.pt",
            {
                "schema_version": 1,
                "signature": signature,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "selection_direction": direction,
                "best_validation_score": best,
                "history": history,
                "model_weights": model_weights(model),
                "best_model_weights": best_weights,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": capture_rng(adapter.device),
            },
        )
        write_json(output / "history.json", history)
        print(
            f"{cfg.kind}: epoch {epoch}, loss={mean_loss:.4f}, validation {metric}={validation['metrics'][metric]:.4f}",
            flush=True,
        )
    return {
        "history": history,
        "best_epoch": best_epoch,
        "selection_metric": metric,
        "selection_direction": direction,
        "best_validation_score": best,
    }
