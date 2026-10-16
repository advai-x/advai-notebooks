"""Native MoE adaptation mechanics and routing diagnostics, not task supervision.

The first allowlist is deliberately narrow: OLMoE and Qwen3-MoE in the pinned SDK.
Base/Instruct still own language targets, metrics, selection and Trainer assembly.
"""

import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback

SUPPORTED = {"olmoe", "qwen3_moe"}
TINY_IDS = {"tiny", "tiny-olmoe", "tiny-qwen3-moe"}


def model_type(config):
    name = config.model_type
    looks_sparse = (
        "moe" in name.lower()
        or name == "mixtral"
        or any(
            getattr(config, key, 0) and getattr(config, key, 0) > 1
            for key in ("num_experts", "num_local_experts")
        )
    )
    if looks_sparse and name not in SUPPORTED:
        raise ValueError("MoE v1 supports only native olmoe and qwen3_moe backends")
    return name if name in SUPPORTED else None


def check_architecture(config, settings, *, supports_moe):
    family = model_type(config)
    if family and not supports_moe:
        raise ValueError("MoE is currently supported only by Base CPT and Instruct SFT, not RL")
    if settings.architecture == "moe" and not family:
        raise ValueError("Requested MoE but the loaded model is dense")
    if settings.architecture == "dense" and family:
        raise ValueError("Requested dense but the loaded model is MoE")
    return family


def weight_budget(config, settings, device="cpu"):
    """Weight-only lower bound before a real sparse checkpoint is allocated/downloaded."""
    if settings.max_weight_memory_gb is None:
        return
    with torch.device("meta"):
        empty = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    count = sum(p.numel() for p in empty.parameters())
    size = torch.tensor([], dtype=getattr(torch, settings.dtype)).element_size()
    gigabytes = count * size / 1e9
    if gigabytes > settings.max_weight_memory_gb:
        raise ValueError(
            f"Weight-only lower bound {gigabytes:.3f} GB exceeds max_weight_memory_gb; "
            "training needs additional gradient/optimizer/activation memory"
        )
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        if count * size > free:
            raise ValueError(
                "Model weights alone exceed currently free CUDA memory; "
                "active expert count does not reduce weight residency"
            )


def validate_adapter_targets(model, training):
    if not model_type(model.config) or training.method != "lora":
        return
    for name in training.lora_target_modules or []:
        if name == "gate" or name.endswith(".gate"):
            raise ValueError("MoE LoRA must not target the frozen router")
    parameters = dict(model.named_parameters())
    for target in training.lora_target_parameters or []:
        matches = [
            (name, p)
            for name, p in parameters.items()
            if name == target or name.endswith("." + target)
        ]
        if not matches:
            raise ValueError(f"No expert parameter matches {target}")
        if any(".mlp.experts." not in name or p.ndim != 3 for name, p in matches):
            raise ValueError("MoE target_parameters must identify native 3D expert weights")


def configure(model, training=None):
    family = model_type(model.config)
    if not family:
        if training and (
            training.lora_target_parameters is not None
            or training.moe_aux_loss_coef is not None
            or training.moe_train_router is not None
        ):
            raise ValueError("MoE training settings require a supported MoE model")
        return
    model.config.output_router_logits = True
    if training is not None:
        if training.moe_aux_loss_coef is not None:
            model.config.router_aux_loss_coef = training.moe_aux_loss_coef
        train_router = (
            training.method == "full"
            if training.moe_train_router is None
            else training.moe_train_router
        )
        if training.method == "lora" and train_router:
            raise ValueError("MoE LoRA freezes routers; choose full to train the router")
        model.config.advai_moe_train_router = train_router
    if not getattr(model.config, "advai_moe_train_router", True):
        for name, parameter in model.named_parameters():
            if ".mlp.gate." in name:
                parameter.requires_grad_(False)
    coefficient = model.config.router_aux_loss_coef
    if isinstance(coefficient, bool) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Native router auxiliary coefficient must be finite and nonnegative")


def coefficient(model):
    return model.config.router_aux_loss_coef if model_type(model.config) else None


def auxiliary(outputs, lm_loss, count, denominator, aux_loss_coef):
    if aux_loss_coef is None:
        return lm_loss
    value = getattr(outputs, "aux_loss", None)
    if value is None or value.ndim != 0 or not torch.isfinite(value):
        raise ValueError("MoE must return a finite scalar aux_loss; enable output_router_logits")
    # A target-token-weighted mean of per-microbatch routing objectives. Never
    # multiply this auxiliary objective by the gradient-accumulation count.
    return lm_loss + aux_loss_coef * value * count / denominator


def describe(model):
    family = model_type(model.config)
    if not family:
        return None
    parameters = list(model.named_parameters())
    return {
        "model_type": family,
        "num_experts": model.config.num_experts,
        "experts_per_token": model.config.num_experts_per_tok,
        "total_parameters": sum(p.numel() for _, p in parameters),
        "trainable_parameters": sum(p.numel() for _, p in parameters if p.requires_grad),
        "weight_bytes": sum(p.numel() * p.element_size() for _, p in parameters),
        "router_trainable": getattr(model.config, "advai_moe_train_router", True),
        "aux_loss_coef": model.config.router_aux_loss_coef,
        "parameter_dtypes": sorted({str(p.dtype) for _, p in parameters}),
        "note": "Actual parameter storage, not peak training memory or active-parameter size.",
    }


class RoutingAccumulator:
    def __init__(self, config):
        self.enabled = bool(model_type(config))
        self.experts = getattr(config, "num_experts", 0)
        self.top_k = getattr(config, "num_experts_per_tok", 0)
        self.layers, self.aux_sum, self.targets = [], 0.0, 0

    def update(self, output, attention_mask, target_count):
        if not self.enabled:
            return
        routers = getattr(output, "router_logits", None)
        value = getattr(output, "aux_loss", None)
        if not routers or value is None or not torch.isfinite(value):
            raise ValueError(
                "MoE evaluation requires native router logits and finite auxiliary loss"
            )
        if not self.layers:
            self.layers = [
                {"tokens": 0, "counts": torch.zeros(self.experts, dtype=torch.long), "entropy": 0.0}
                for _ in routers
            ]
        if len(routers) != len(self.layers):
            raise ValueError("MoE router layer count changed during evaluation")
        mask = attention_mask.reshape(-1).bool()
        for logits, layer in zip(routers, self.layers):
            flat = logits.reshape(-1, self.experts)
            if len(flat) != len(mask):
                raise ValueError("Unsupported MoE router-logit layout")
            probs = flat[mask].float().softmax(-1)
            if not torch.isfinite(probs).all():
                raise ValueError("Nonfinite routing probabilities")
            choices = probs.topk(self.top_k, dim=-1).indices
            layer["counts"] += torch.bincount(choices.flatten(), minlength=self.experts).cpu()
            layer["tokens"] += len(probs)
            layer["entropy"] += float(-(probs * probs.clamp_min(1e-30).log()).sum())
        self.aux_sum += float(value) * int(target_count)
        self.targets += int(target_count)

    def report(self):
        if not self.enabled:
            return None
        result = []
        for index, layer in enumerate(self.layers):
            assignments = layer["tokens"] * self.top_k
            counts = layer["counts"].tolist()
            if sum(counts) != assignments or not assignments:
                raise ValueError("Router assignment counts do not reconcile")
            result.append(
                {
                    "index": index,
                    "input_tokens": layer["tokens"],
                    "assignment_counts": counts,
                    "assignment_share": [count / assignments for count in counts],
                    "max_to_mean_load": max(counts) * self.experts / assignments,
                    "mean_router_entropy_nats": layer["entropy"] / layer["tokens"],
                }
            )
        return {
            "aux_loss": self.aux_sum / self.targets,
            "layers": result,
            "note": "Routing covers all nonpadding input tokens, including SFT prompts. "
            "Shares divide top-k assignments, not unique tokens; layer indices are MoE order. "
            "Auxiliary loss is microbatch-local and target-token weighted; it is not perplexity.",
        }


class GradientAudit(TrainerCallback):
    """Checkpointed post-clipping gradient diagnostics without RNG or optimizer changes."""

    def __init__(self, record):
        self.record = record
        self.record.setdefault("moe_gradients", {"optimizer_steps": 0, "groups": {}})

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        audit = self.record["moe_gradients"]
        audit["optimizer_steps"] += 1
        groups = {
            name: {"trainable_tensors": 0, "nonzero_tensors": 0, "squared_norm": 0.0}
            for name in ("router", "experts", "other")
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            key = (
                "router" if ".mlp.gate." in name else "experts" if ".experts." in name else "other"
            )
            group = groups[key]
            group["trainable_tensors"] += 1
            if parameter.grad is not None:
                norm = 0.0
                for chunk in parameter.grad.detach().reshape(-1).split(1_000_000):
                    gradient = chunk.float()
                    if not torch.isfinite(gradient).all():
                        raise ValueError("Nonfinite MoE gradients")
                    norm += float(gradient.square().sum())
                group["squared_norm"] += norm
                group["nonzero_tensors"] += int(norm > 0)
        for name, current in groups.items():
            previous = audit["groups"].setdefault(name, {"nonzero_gradient_steps": 0})
            previous.update(
                trainable_tensors=current["trainable_tensors"],
                last_nonzero_tensors=current["nonzero_tensors"],
                last_post_clip_grad_norm=math.sqrt(current["squared_norm"]),
            )
            previous["nonzero_gradient_steps"] += int(current["nonzero_tensors"] > 0)


class NativeCheckpointTrainer(Trainer):
    """Keep native MoE checkpoint conversion, then let Trainer restore optimizer/RNG.

    Transformers 5.18 saves fused experts in their interoperable per-expert layout.
    Its Trainer raw state-dict loader does not run that layout conversion. The task
    pipeline therefore restores via from_pretrained before constructing the optimizer.
    This guard skips only that redundant model reload, not any other resume state.
    """

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        if model_type(target.config) and not hasattr(target, "peft_config"):
            restored = getattr(target, "_advai_loaded_checkpoint", None)
            if restored != str(Path(resume_from_checkpoint).resolve()):
                raise ValueError("Restore the native MoE checkpoint before creating Trainer")
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
