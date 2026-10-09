# VLM — Vision-Language Model

English | [简体中文](vlm.zh-CN.md)

This track implements native Hugging Face LLaVA single-image question answering:
image + question → text answer. It supports full fine-tuning and language-side
`q_proj`/`v_proj` LoRA. Full mode can freeze or update the vision tower; LoRA updates
only language adapters and requires `freeze_vision: true`. This is a specific backend,
not generic support for every VLM architecture.

## Quick start

```bash
python -m pip install -e '.[vlm,notebooks,dev]'
python -m advai_notebooks.vlm validate --config configs/vlm/smoke.yaml
python -m advai_notebooks.vlm run --config configs/vlm/smoke.yaml
# A separate tiny LoRA experiment with its own output directory
python -m advai_notebooks.vlm run --config configs/vlm/lora.yaml
```

The [notebook](../notebooks/vlm/01_finetune.ipynb) includes an executed CPU example
with saved outputs. The default `model.id: tiny` is a randomly initialized LLaVA
(small Llama decoder + CLIP vision tower). It checks the training/export pipeline;
its generated text does not demonstrate a usable visual assistant.

## Data contract

Use UTF-8 JSONL with one single-image question/answer per line:

```json
{"id":"scene-1-qa","group_id":"scene-1","split":"train","image":"images/scene-1.png","prompt":"What color is the square?","answer":"Red."}
```

- `train`, `validation` and `test` must all be nonempty. IDs must be unique; groups cannot cross splits.
- `image` is relative to the JSONL directory. Absolute paths, `..` and escaping symlinks are rejected; images must decode to RGB.
- The same decoded image cannot cross splits. Assign related scenes, near-duplicates and augmentations to the same group yourself.
- `prompt` and `answer` must be nonempty. Encoding uses the native processor's chat template and image token, not text-only LLM batching.
- Prompt tokens (including visual tokens) and padding receive labels of `-100`. Only the answer and EOS are supervised.
- `max_len` includes visual tokens. Oversized inputs fail rather than silently truncating images or answers; inference also reserves `max_new_tokens`.
- The schema supports one user image question and one assistant answer, not multiple images, video or multi-turn conversations.

## Training, selection and recovery

`validation.nll` is weighted by valid answer tokens; lower is better. Reports also
include perplexity. The epoch-zero baseline can win model selection. Test data is
used only for final reporting, never selection. Gradient accumulation is weighted
by valid answer tokens so answer lengths and partial final batches do not change
the loss definition.

```bash
python -m advai_notebooks.vlm run --config configs/vlm/smoke.yaml \
  --resume outputs/vlm/smoke/latest
python -m advai_notebooks.vlm evaluate --config configs/vlm/smoke.yaml \
  --checkpoint outputs/vlm/smoke/export --split test --output outputs/vlm/test.json
```

Recovery restores the latest completed epoch, optimizer and RNG state. Configuration,
data/media fingerprints, source code and dependency versions must match the original
run. The first backend is single-process, single-device fp32; QLoRA, quantized bases,
AMP and distributed training are not supported.

## Inference and export

```python
from advai_notebooks.vlm.model import VLMModel

model = VLMModel.load("outputs/vlm/smoke/export")
answer = model.chat("data/sample/vlm/images/cyan.ppm", "what color is the square ?")
```

CLI `chat --checkpoint ... --input requests.json --output responses.json` accepts a
JSON array, for example:

```json
[{"image":"data/sample/vlm/images/cyan.ppm","prompt":"what color is the square ?"}]
```

Inference image paths are relative to the current working directory.

`export/` contains the validation-selected model; `latest_export/` contains the
final epoch. Full exports save native weights, config, processor, tokenizer and chat
template. LoRA exports save the adapter and a bundled base, without depending on the
original local base path after publication. Bundling increases storage per checkpoint.

```bash
python -m advai_notebooks.vlm export --checkpoint outputs/vlm/lora/export \
  --output outputs/vlm/merged --merged
python -m advai_notebooks.vlm publish --export-dir outputs/vlm/merged \
  --repo-id your-org/your-vlm --dry-run
```

Publication selects and reloads inference files; it excludes raw images, data,
optimizer state and logs. Only explicitly removing `--dry-run` enables Hub access.
Check model and dataset licenses before publishing.

## Real bases and validation limits

The [revision-pinned LLaVA template](../configs/vlm/hf.yaml) uses
`llava-hf/llava-1.5-7b-hf`. Its processor, visual token count and answer boundaries
have been checked separately; **7B GPU fine-tuning has not been run**. Seven billion
fp32 parameters alone require roughly 28 GB, before additional components, gradients,
optimizer state and activations. This is not a free-tier Colab preset.

Replace the JSONL and output paths, and choose a compatible native `model_type=llava`
base for your hardware budget. Local native LLaVA checkpoints are also valid
`model.id` values; Hub bases require a revision. `validate` loads the complete base
and processor to check actual input budgets, so validating a real configuration
also requires sufficient memory. Use smoke first when target hardware is unavailable.

Offline tests cover full/LoRA training, answer masks, merged exports, processor
save/reload, strict recovery and publication previews. NLL is not QA accuracy;
real experiments need answer correctness, grounding and human review. Qwen-VL,
LLaVA-OneVision and video backends are unsupported. CUDA/MPS and real-domain
training remain unverified.

Official references: [LLaVA](https://huggingface.co/docs/transformers/model_doc/llava),
[multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal).
