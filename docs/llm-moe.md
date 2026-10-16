# Native MoE training and fine-tuning

English | [简体中文](llm-moe.zh-CN.md)

Mixture of Experts (MoE) is a decoder **architecture**, not a new training objective.
Use the existing Base task for continued pre-training (CPT) and Instruct for
assistant-only SFT. Both retain their own data, loss, metrics and selection logic.
The first supported native backends are `olmoe` and `qwen3_moe`; other sparse
architectures fail explicitly instead of silently dropping routing objectives.

Start with the executable [Base tutorial](../notebooks/llm/base/02_moe_finetune.ipynb)
or [Instruct tutorial](../notebooks/llm/instruct/02_moe_finetune.ipynb):
[Base in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/base/02_moe_finetune.ipynb) ·
[Instruct in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/instruct/02_moe_finetune.ipynb).
These are companion architecture tutorials, not additional task tracks.

## Install and choose a configuration

```bash
python -m pip install -e '.[llm-base,llm-instruct,notebooks,dev,publish]'
python -m advai_notebooks.llm.base validate --config configs/llm/base/moe/smoke.yaml
python -m advai_notebooks.llm.base run --config configs/llm/base/moe/smoke.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/moe/smoke.yaml
```

Verified SDK pins: Transformers 5.18.0, PEFT 0.18.1, Accelerate 1.15.0.
Each `configs/llm/{base,instruct}/moe/` directory provides:

| Configuration | Model and update policy | Purpose |
| --- | --- | --- |
| `smoke.yaml` | Random native tiny OLMoE; full weights, router trainable | Offline CPU lifecycle |
| `smoke_qwen3.yaml` | Random native tiny Qwen3-MoE; full weights | Second native backend |
| `smoke_lora.yaml` | Tiny OLMoE attention q/v LoRA; router frozen | Small adapter lifecycle |
| `smoke_experts_lora.yaml` | Tiny attention plus native 3D expert-parameter LoRA; router frozen | Expert gradients and merge/reload |
| `olmoe_lora.yaml` | Pinned OLMoE Base or Instruct; attention LoRA | Real-model GPU template |
| `qwen3_lora.yaml` | Pinned Qwen3-30B-A3B Base or post-trained model; attention LoRA | Real-model GPU template |

Tiny models have four experts, top-2 routing and a small fixed synthetic tokenizer.
They are actual sparse models, not dense stand-ins, but are **not pretrained models**.
All smoke data is synthetic; successful execution is not evidence of domain quality.

## Data and architecture contract

Keep the [existing Base/Instruct data contracts](llm.md#data-contracts): Base JSONL
contains grouped documents; Instruct contains text-only conversations. Split original
documents/conversations before encoding. All train/validation/test splits must exist;
test never chooses weights. Base handles boundary-preserving document chunks;
Instruct masks system/user/role prefixes and supervises assistant replies/end markers.

`model.architecture` is `auto` (detect), `dense` or `moe`. Explicit mismatches fail.
`model.id: tiny-olmoe` and `tiny-qwen3-moe` select offline native backends.
Hub MoE initialization requires a 40-character immutable revision SHA and disables
remote code. Local full/merged exports are also accepted. Adapter-only exports must
be merged before initializing a different training task.

The real OLMoE Instruct preset uses its pinned native chat template. The Qwen3
Instruct preset deliberately supplies an explicit **text-only non-thinking** Jinja
template: Qwen role/end tokens, an empty `<think>` prefill outside supervision, and
generation blocks around assistant content/end markers. This makes assistant masks
and inference prefixes stable across turns. It is not a thinking/reasoning SFT
recipe or a replacement for every upstream dynamic-template feature. Inspect token
IDs, decoded labels and inference prompts when changing templates. Pinned real
tokenizers were checked on a multi-turn example; real-weight training was not run.

## Update policy and LoRA targets

Full training defaults to a trainable router. `training.moe_train_router: false`
explicitly freezes router weights while other full-model parameters remain trainable.
LoRA defaults to a frozen router; `moe_train_router: true` with LoRA is rejected in
v1. This is a defined scope limitation, not router-LoRA training support.

```yaml
model:
  id: tiny-olmoe
  architecture: moe
training:
  method: lora
  moe_train_router: false
  moe_aux_loss_coef: 0.01
  lora_rank: 4
  lora_alpha: 8
  lora_target_modules: [q_proj, v_proj]
  lora_target_parameters:
    - mlp.experts.gate_up_proj
    - mlp.experts.down_proj
```

In the pinned SDK these expert weights are three-dimensional `nn.Parameter`s,
not ordinary Linear modules. PEFT `target_parameters` must be used; a generic
attention-only `target_modules` list does not adapt experts. Exact/suffix matches
must resolve to native 3D expert weights or initialization fails. For expert-only
LoRA, use `lora_target_modules: []` with the parameter targets. Router module
targets are rejected. Inspect `validate` output and `training.moe_gradients` to
confirm what actually trains, rather than inferring it from the word “LoRA”.

PEFT 0.18.1 emits a generic unsupported-layer warning for these expert parameter
wrappers. CPU tests exercise both tiny native families, verify nonzero expert-adapter
gradients, unchanged frozen base weights and merged/reloaded logits. They do not
establish compatibility with every real checkpoint, SDK version or CUDA kernel.
Real presets intentionally default to attention-only LoRA; enable expert targets
only after a bounded target-hardware gradient/merge check.

## Objective, evaluation and diagnostics

Native router logits are enabled and their load-balancing auxiliary loss participates
in training. `moe_aux_loss_coef` overrides the native coefficient; leaving it null
keeps the native model value. Zero disables its contribution, but native finite
router outputs are still required for diagnostics. No extra router z-loss is added.

For microbatch `i` in one gradient-accumulation window:

```text
loss_i = shifted_CE_sum_i / N_window
       + coefficient * native_aux_i * N_targets_i / N_window
N_window = sum_i N_targets_i
```

This is a target-token-weighted average of **microbatch-local** auxiliary losses.
There is no second division/multiplication by the accumulation count. Load balancing
is nonlinear: changing microbatch partitioning can change this objective; it is
not identical to computing router loss on one concatenated large batch.

Selection remains validation **NLL**, including epoch zero. Test NLL and perplexity
exclude routing loss; PPL = exp(NLL). Keep tokenizer, corpus and supervision region
compatible before comparing perplexities.

Reports add a separate `moe` section containing target-weighted native auxiliary
loss and per-MoE-layer assignment counts/shares, max-to-mean load and mean router
entropy in nats. Padding is excluded; routing covers **all nonpadding input tokens**,
including SFT prompts, not just supervised reply tokens. Counts sum to input tokens
times top-k; shares sum to one and divide assignments, not unique tokens. Layer
indices follow native MoE-layer order. These diagnostics can expose collapse but
balanced routing alone does not prove useful expert specialization or better quality.

`training.moe_model` records actual parameter counts, dtypes and parameter bytes.
`training.moe_gradients` records optimizer-step counts, trainable tensors, nonzero
gradient steps and last post-clipping norms for router/experts/other parameters.
Audits do not retain gradient tensors or modify optimizer/RNG state.

## Resume, export and inference

```bash
python -m advai_notebooks.llm.base run --config configs/llm/base/moe/smoke.yaml \
  --resume outputs/llm/base/moe-smoke/latest
python -m advai_notebooks.llm.instruct export \
  --checkpoint outputs/llm/instruct/moe-smoke-experts-lora/export \
  --output outputs/llm/instruct/moe-merged --merged
```

Use the exact `output_dir` from your configuration. Complete epoch checkpoints
include model, optimizer, scheduler, RNG and the experiment signature; inference
exports do not. The pinned SDK saves full MoE expert weights in an interoperable
per-expert layout. Recovery first loads through the native model converter, then
Trainer restores optimizer/scheduler/RNG without incorrectly raw-loading expert keys.
CPU/fp32 interrupted/uninterrupted tests verify identical histories, gradient audits
and final weights for full and expert LoRA on both supported tiny families.

Full and merged exports load with `AutoModelForCausalLM.from_pretrained`; adapters
use PEFT and pinned Hub bases or bundled local/tiny bases. Metadata retains dtype,
architecture, auxiliary coefficient and router policy. Merge reloads a copy and
does not modify the active adapter. Generation disables unnecessary routing outputs;
native generation config and the training/inference template are preserved.
Use existing `generate`/`chat`, `evaluate`, `export` and offline `publish --dry-run`
commands from the [LLM guide](llm.md). Publication is explicit, never part of training.

## Real-model memory and scope

Copy the real preset into ignored `configs/local/`, keep the repository root explicit
with `--root .`, replace private data/output paths, and review model/data licenses.
Do not train real checkpoints on the tiny synthetic fixtures. Hub revisions in the
presets pin model and tokenizer; changing a revision is a new experiment.

`model.dtype` controls actual weight loading; `runtime.precision` controls AMP.
The GPU templates explicitly use `bfloat16` weights and `bf16` AMP. Reduced active
parameters do **not** mean only those weights reside in memory. OLMoE-1B-7B needs
roughly 14 GB and Qwen3-30B-A3B roughly 61 GB for bf16 weights alone (decimal GB),
before adapters, gradients, optimizer state, logits, activations and temporary buffers.
Merge/export also needs sufficient host memory.

`max_weight_memory_gb` checks a native meta-model parameter-count lower bound before
real weight loading, and rejects weights exceeding currently free CUDA memory.
It is not a training-peak estimator or admission guarantee. Real `validate` loads
weights too; it is not a cheap metadata-only probe. A rejected cap is not permission
to blindly raise it. Neither preset is guaranteed to fit free Colab; Qwen3 is not
a small single-GPU example just because its active parameter count is smaller.

Implemented and locally verified: native tiny CPT/SFT, full/attention/expert LoRA,
router auxiliary objective and diagnostics, exact CPU epoch recovery, native
export/reload/merge, inference and offline publication. Real-tokenizer formatting
checks are separate from training tests. Real pretrained GPU training, CUDA/AMP
recovery and domain benchmarks remain **unverified**. No distributed/expert-parallel
training, quantization/QLoRA, router LoRA, large-scale pretraining from scratch or
MoE PPO/GRPO is provided in this first version. Repository MIT licensing does not
replace upstream model/dataset licenses.

Official references: [OLMoE](https://huggingface.co/docs/transformers/model_doc/olmoe),
[Qwen3-MoE](https://huggingface.co/docs/transformers/model_doc/qwen3_moe),
[PEFT parameter targeting](https://huggingface.co/docs/peft/package_reference/lora#targeting-nnparameter-directly),
[OLMoE model card](https://huggingface.co/allenai/OLMoE-1B-7B-0924),
[Qwen3 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B).
