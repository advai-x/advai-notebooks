# LLM Base and Instruct fine-tuning

English | [简体中文](llm.zh-CN.md)

These independent tracks implement domain continued pre-training (Base) and
instruction supervised fine-tuning (Instruct SFT). Each owns its configuration,
data, sampling, loss, optimizer, training, evaluation, metrics, model facade,
CLI and publication modules. They use native Transformers decoder/LM heads and
Trainer, with PEFT for LoRA. Neither imports another task.

## Install and run

```bash
python -m pip install -e '.[llm-base,llm-instruct,notebooks,dev]'
python -m advai_notebooks.llm.base validate --config configs/llm/base/smoke.yaml
python -m advai_notebooks.llm.base run --config configs/llm/base/smoke.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/smoke.yaml
```

Command aliases are `advai-llm-base` and `advai-llm-instruct`. Each track provides
`smoke.yaml`, `lora.yaml` and `hf.yaml`. The real-model template requires your own
data; it is not a completed domain-quality experiment. Installing either task extra
alone is sufficient for that task. Verified SDK versions are Transformers 5.18.0,
PEFT 0.18.1 and Accelerate 1.15.0.

For native OLMoE / Qwen3-MoE CPT and SFT, see the [MoE guide](llm-moe.md)
and companion Base/Instruct notebooks. MoE is an architecture within these tasks,
not a separate task; dense configurations remain unchanged.

## Data contracts

Base JSONL:

```json
{"id":"doc-1","document_id":"book-1","text":"The sky is blue.","split":"train"}
```

Instruct JSONL:

```json
{"id":"conversation-1","conversation_id":"session-1","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What color is the sky?"},{"role":"assistant","content":"Blue."}],"split":"train"}
```

`prompt`/`completion` can replace `messages`; they are normalized to user/assistant
turns. The first implementation supports text-only conversations with an optional
initial system message followed by alternating user/assistant turns. Training
conversations end with assistant; inference requests end with user. Tool calls and
multimodal messages are unsupported.

All three splits (`train`, `validation`, `test`) must be nonempty. Split by document
or conversation before encoding and chunking. Duplicate IDs, groups spanning splits
and identical content across splits are rejected. Express shared provenance through
group IDs; validators do not detect semantically paraphrased duplicates.

Base appends EOS to every document. With `reject_truncation: false`, long documents
are sliced with one overlapping context token, preserving every boundary target
exactly once and never joining different documents. Set it to true to reject long
documents. Padding is masked by position, so real EOS remains supervised even when
pad and EOS share an ID.

Instruct supervises only assistant regions. It prefers template `{% generation %}`
blocks and fast-tokenizer offsets; trailing whitespace must not consume a subsequent
user turn. Without generation blocks, it checks rendered-prefix stability before
locating reply regions. Unstable prefixes and boundary-spanning tokens are rejected;
provide a suitable Jinja `model.chat_template`. Templates must match the base
vocabulary, objective and inference format; do not arbitrarily replace role markers.
The default `reject_truncation: true` rejects oversized conversations. Explicit false
enables right truncation, records dropped tokens and requires at least one supervised
reply token. Review conversation lengths to avoid cutting answers or end markers.

## Training, evaluation and recovery

`training.method` is `full` or `lora`. Configure optimizer, learning rate, weight
decay, epochs, batch size, gradient accumulation/clipping, gradient checkpointing,
seed and LoRA rank/alpha/dropout/target modules. This implementation uses a constant
learning rate and single-device Trainer; it does not automatically enable multiple
GPUs, quantization or model sharding. CPU/MPS use fp32; CUDA can use fp16/bf16 when
supported by the target hardware.

The lifecycle is data/template validation → baseline validation/test → training →
per-epoch validation → validation-NLL selection (including epoch zero) → test
comparison → export. Test never affects training or selection. NLL is weighted by
valid shifted supervision tokens, not an average of batch-average losses.
PPL = exp(NLL); compare only compatible tokenizers, corpora and supervision regions.

Each task independently implements metrics, metric records and selection.
Instruct optionally supports `generation_metrics: [exact_match, json_validity]`,
counted by conversation. Generation evaluation is always greedy, independent of
inference sampling settings. Low NLL alone does not demonstrate instruction-following
ability in a business domain.

```text
output_dir/
  config.json, environment.json, data_summary.json
  baseline/                         # Includes the epoch-zero model
  baseline_validation.json, baseline_test.json
  trainer/checkpoint-N/             # Native model + optimizer/scheduler/RNG
    advai_metadata.json             # Inference facade metadata
    advai_resume.json               # Signature, history, best checkpoint pointer
  latest.json                       # Latest complete checkpoint
  best/                             # Validation-selected inference model
  export/                           # Deployment/publication inference model
  report.json
```

```bash
python -m advai_notebooks.llm.base run --config configs/llm/base/smoke.yaml \
  --resume outputs/llm/base/run/latest
```

`latest` is a logical alias resolved through `latest.json`; a full
`trainer/checkpoint-N` path also works. Inference exports omit optimizer/scheduler/RNG
and cannot resume training. Recovery requires unchanged planned epochs, configuration,
data, packages, device, tokenizer/template and relevant source fingerprints.
Use a new output directory for a changed experiment. `advai_resume.json` marks a
complete checkpoint; incomplete epochs replay from the previous saved boundary.
CPU/fp32 tests verify matching interrupted/uninterrupted full/LoRA weights.
CUDA/MPS/AMP recovery needs target-hardware validation.

## Models, inference and optional handoff

```python
from advai_notebooks.llm.base.model import BaseModel
from advai_notebooks.llm.instruct.model import InstructModel

base = BaseModel.load("outputs/llm/base/run/export")
texts = base.generate(["The sky is"], max_new_tokens=8)

instruct = InstructModel.load("outputs/llm/instruct/run/export")
answer = instruct.chat([{"role": "user", "content": "What color is the sky?"}], max_new_tokens=8)
```

`generate` returns ordered continuations; `chat` returns one reply string. Neither
includes the input prompt. Generation is greedy by default, with configurable
`do_sample`, `temperature`, `top_p` and `max_new_tokens`.
CLI `generate --input` accepts a string list; `chat --input` accepts a conversation
list (each conversation is a messages array). Both write `{"outputs": [...]}` to
`--output`. Prompt plus generation budget must fit `model.max_len`; inputs are not
silently truncated. Full models retain native generation config; facade settings
are saved in `advai_metadata.json`.

Base → Instruct is an optional artifact handoff. Point Instruct `model.id` at a
Base full/merged export and confirm or supply a compatible chat template.
Adapter-only directories cannot initialize training; merge first. Handoff uses
native files, not imports from the Base task.

## Full, adapter and merged exports

Full exports save native weights/config, tokenizer/template, generation config and
facade metadata, and support `AutoModelForCausalLM.from_pretrained`. LoRA exports
save adapters and tokenizer/template with base provenance. Hub bases are pinned
to resolved commit SHAs; tiny/local bases include a frozen `base_model/` snapshot.
Bundling enables relocation but can make an adapter package large. Prefer the model
facade for bundled adapters; native PEFT requires explicitly loading `base_model/` first.

```bash
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/lora.yaml
python -m advai_notebooks.llm.instruct export \
  --checkpoint outputs/llm/instruct/lora-run/export \
  --output outputs/llm/instruct/merged-export --merged
```

Safe merge reloads a copy without modifying the active model. Merged exports load
independently of PEFT adapters. Quantized adapters and QLoRA merging are unsupported.

## Hugging Face publication

```bash
python -m advai_notebooks.llm.base publish \
  --export-dir outputs/llm/base/run/export \
  --repo-id your-org/your-llm-base --dry-run \
  --output outputs/publishing/llm-base-plan.json
python -m advai_notebooks.llm.instruct publish \
  --export-dir outputs/llm/instruct/run/export \
  --repo-id your-org/your-llm-instruct --dry-run \
  --output outputs/publishing/llm-instruct-plan.json
```

Publication defaults to private; remove `--dry-run` explicitly to upload. Training
and notebooks do not upload by default. Packages contain native inference assets,
English model cards and SHA256 manifests, excluding raw conversations, training
state, logs and credentials. Adapters retain tokenizer, chat template, pinned base
contracts and any bundled base. Offline preview uses only cached external bases:
missing caches fail without downloading. Hub loading supports `revision`, `cache_dir`
and `local_files_only`; pin a publication SHA for deployment.

Recorded Base/Instruct demo releases are listed in [Datasets](datasets.md).
Those synthetic fixtures are not domain-quality checkpoints.

## Current scope

Implemented: Base CPT, Instruct SFT, full/LoRA, single-device training, native
export/reload, validation selection, epoch recovery, CLI, notebooks and publication.
Tiny offline data is for functional checks. QLoRA, cross-document packing,
distributed training, streaming, DPO and domain benchmarks remain future work.
PPO/GRPO are independent tasks, not Instruct SFT; see the [RL guide](rl.md).

Official references: [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer),
[chat templates](https://huggingface.co/docs/transformers/chat_templating),
[PEFT checkpoints](https://huggingface.co/docs/peft/developer_guides/checkpoint).
