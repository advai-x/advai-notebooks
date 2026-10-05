"""Single-device native decision fine-tuning; resumable checkpoints at epoch boundaries."""

import math

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.checkpoint import (
    capture_rng,
    model_weights,
    restore_rng,
    save_training_state,
)

from .evaluate import evaluate
from .losses import build_objective
from .metrics import is_improvement, select_metric
from .model import DecisionModel
from .optimizer import create_optimizer
from .runtime import amp_context, forward, make_batch
from .sampling import decision_groups


def train_model(cfg, model, train_items, validation_items, device, signature, resume_state=None):
    import torch

    output = cfg.path(cfg.output_dir)
    tc = cfg.training
    objective = build_objective(tc)
    model.to(device).float()
    if cfg.model.freeze_encoder:
        model.encoder.requires_grad_(False)
    if tc.gradient_checkpointing:
        if not cfg.model.freeze_encoder:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    optimizer = create_optimizer(model, tc)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.runtime.precision == "fp16")
    metric = cfg.evaluation.selection_metric
    direction = "maximize" if cfg.evaluation.maximize else "minimize"
    maximize = direction == "maximize"
    first_epoch, best_score, best_epoch, history = 0, -math.inf if maximize else math.inf, 0, []
    best_weights = None
    if resume_state is not None:
        state = resume_state
        if state["signature"] != signature:
            raise ValueError(
                "Resume configuration/data/implementation differ from the saved experiment"
            )
        first_epoch, best_score, best_epoch, history = (
            state["epoch"],
            state["best_validation_score"],
            state["best_epoch"],
            state["history"],
        )
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_weights = state["best_model_weights"]
        model.load_state_dict(best_weights)
        model.save(output / "best")
        model.load_state_dict(state["model_weights"])
        restore_rng(state["rng"], device)
    else:
        # Old calibration does not describe updated parameters.
        model.config["temperature"] = [1.0] * 3
        model.config.pop("temperature_by_options", None)
        if cfg.evaluation.include_baseline:
            baseline, _ = evaluate(
                model, {"validation": validation_items}, cfg, "validation", calibrated=False
            )
            selected = select_metric(baseline, metric)
            if selected["direction"] != direction:
                raise ValueError("Selection metric direction differs from configuration")
            best_score = selected["value"]
            best_weights = model_weights(model)
            model.save(output / "best")
            history.append({"epoch": 0, "train_loss": None, "validation": baseline})

    for epoch in range(first_epoch, tc.epochs):
        model.train()
        if cfg.model.freeze_encoder:
            model.encoder.eval()
        sigma = tc.sigma_start + (tc.sigma_end - tc.sigma_start) * epoch / max(1, tc.epochs - 1)
        epoch_loss = 0.0
        for group in decision_groups(train_items, tc, epoch):
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(group), tc.batch_size):
                chunk = group[micro_start : micro_start + tc.batch_size]
                batch = make_batch(chunk, model.tokenizer.pad_token_id, device)
                with amp_context(device, cfg.runtime.precision):
                    logits = forward(model, batch)
                    loss = objective(
                        logits, batch["target"], batch["option_mask"], batch["qtype"], sigma
                    )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; check data and precision")
                # Correctly scale the final partial accumulation group and uneven microbatches.
                scaler.scale(loss * len(chunk) / len(group)).backward()
                epoch_loss += float(loss.detach()) * len(chunk)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                tc.max_grad_norm,
                error_if_nonfinite=not scaler.is_enabled(),
            )
            scaler.step(optimizer)
            scaler.update()

        metrics, _ = evaluate(
            model, {"validation": validation_items}, cfg, "validation", calibrated=False
        )
        entry = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_items),
            "validation": metrics,
        }
        history.append(entry)
        selected = select_metric(metrics, metric)
        if selected["direction"] != direction:
            raise ValueError("Selection metric direction changed during training")
        score = selected["value"]
        improved = is_improvement(score, best_score, direction)
        if improved:
            best_score, best_epoch = score, epoch + 1
            best_weights = model_weights(model)
            model.save(output / "best")
        latest = model.save(output / "latest")
        state = {
            "signature": signature,
            "epoch": epoch + 1,
            "schema_version": 1,
            "best_validation_score": best_score,
            "best_model_weights": best_weights,
            "best_epoch": best_epoch,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            # Atomic training state keeps weights aligned with optimizer/RNG,
            # even when a later epoch is interrupted during inference export.
            "model_weights": model_weights(model),
            "rng": capture_rng(device),
        }
        save_training_state(latest / "training_state.pt", state)
        write_json(output / "history.json", history)
        print(
            f"epoch={epoch + 1} train_loss={entry['train_loss']:.4f} validation_{metric}={score:.4f}"
        )
    if not (output / "best" / "model.safetensors").is_file():
        raise ValueError(
            "Best checkpoint is missing; resume requires the original experiment directory"
        )
    return DecisionModel.load(output / "best", device), {
        "best_epoch": best_epoch,
        "history": history,
        "selection_metric": metric,
        "selection_direction": direction,
        "best_validation_score": best_score,
    }
