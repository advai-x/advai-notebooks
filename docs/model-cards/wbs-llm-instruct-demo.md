---
library_name: transformers
pipeline_tag: text-generation
tags:
- advai-notebooks
- llm-instruct
- demo
- synthetic-data
- random-initialization
- full
- llama
language:
- en
datasets:
- widebluesky/wbs-llm-instruct-demo
---

# widebluesky/wbs-llm-instruct-demo

## Demo release v1

This is a **trained tiny demo model** for testing training, export, publication,
download and inference. A tiny Llama architecture was initialized randomly
and trained for **two epochs** on synthetic fixtures from `advai-notebooks`.
It is not derived from Meta Llama weights or another pretrained checkpoint.
It is a workflow placeholder for future domain-trained releases.
The model has no demonstrated general language or instruction-following capability.
Generation may be repetitive or incorrect. Synthetic metrics do not establish domain quality.

## Model and training

| Setting | Value |
| --- | --- |
| Architecture | LlamaForCausalLM; random initialization |
| Parameters | 13,248 |
| Layers / hidden size / FFN size | 1 / 32 / 64 |
| Attention heads / key-value heads | 4 / 2 |
| Vocabulary / context length | 123 tokens / 128 tokens |
| Training objective | assistant-only next-token supervision (instruction SFT) |
| Method | Full-parameter training; not a LoRA adapter |
| Epochs completed / selected | 2 / 2 |
| Selection | Lowest validation token-weighted NLL, including epoch zero; test is report-only |
| Dataset | 8 training, 2 validation and 2 test synthetic examples |
| Optimizer / learning rate / weight decay | AdamW / 0.003 / 0.01 |
| Batch size / gradient accumulation | 2 / 2 |
| Scheduler / gradient clipping | Constant / 1.0 |
| Seed / device / precision | 42 / CPU / fp32 |
| Export | Standalone full safetensors model |
| SDK versions | Transformers 5.18.0; PyTorch 2.14.1 |

Only assistant reply tokens are supervised. System and user tokens, and padding,
are masked from the loss. The first training fixture includes multiple assistant turns.
This release is independently trained from random initialization; it does not load
the separately published Base CPT checkpoint.

## Synthetic held-out test metrics

| Metric | Before training | Selected trained export |
| --- | ---: | ---: |
| nll | 4.861312 | 4.737435 |
| perplexity | 129.193582 | 114.141093 |
| exact_match | 0.000000 | 0.000000 |

NLL and perplexity are computed over each task's supervised tokens.
Base and Instruct perplexity values are not directly comparable because their
supervision ranges differ. These results use only two synthetic test examples.

Held-out exact match is 0/2; this release does not demonstrate usable answer quality.

## Inference

In a checkout of `advai-notebooks`:

```bash
python -m pip install -e '.[llm-instruct]'
```

```python
from advai_notebooks.llm.instruct.model import InstructModel

model = InstructModel.load("widebluesky/wbs-llm-instruct-demo", revision="demo-v1", device="cpu")
response = model.chat(
    [{"role": "system", "content": "answer only"},
     {"role": "user", "content": "what color is the sky ?"}],
    max_new_tokens=8,
)
print(response)
```

The full model also supports native Hugging Face loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "widebluesky/wbs-llm-instruct-demo"
model = AutoModelForCausalLM.from_pretrained(repo_id, revision="demo-v1")
tokenizer = AutoTokenizer.from_pretrained(repo_id, revision="demo-v1")
```

For Instruct inference, render the exported chat template with an assistant generation
prompt before passing tokenized inputs to the native model.
Pin the publication commit SHA in `revision` for reproducible deployment.

## Verification and package contents

- All 11 stored weight tensors changed from the random baseline.
- Exported weights exactly match the validation-selected trained checkpoint.
- Fresh-process reload reproduced test metrics exactly.
- Native Hugging Face loading and project facade loading produced identical logits.
- CLI generation/chat and facade API outputs matched.
- File sizes and SHA256 hashes are recorded in `publish_manifest.json`.

The package contains inference weights, architecture and generation configuration,
tokenizer, chat template, task facade metadata, this English model card and a manifest.
Raw data, optimizer state, resumable training checkpoints, logs and credentials are excluded.

Source implementation commit: `f43a44e2431bae9d0451b445cc3bde5f1e2b8493`.

## Training and evaluation dataset

Dataset: [widebluesky/wbs-llm-instruct-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-instruct-demo).
Verified dataset revision: [`21213b40998690495308dac1dae09d669846579b`](https://huggingface.co/datasets/widebluesky/wbs-llm-instruct-demo/tree/21213b40998690495308dac1dae09d669846579b) (`demo-v1`).

This is the synthetic source dataset used for this trained demo release.
Its record fingerprint matches the saved training-run provenance; the published
dataset was reloaded and checked record-by-record before associating it with this model.

The `instruct` configuration contains 8 train, 2 validation and 2 test records.
Original document/conversation group IDs and split assignments are preserved.

```python
from datasets import load_dataset

data = load_dataset(
    "widebluesky/wbs-llm-instruct-demo", "instruct",
    revision="21213b40998690495308dac1dae09d669846579b", token=True,
)
```

Access uses the existing Hugging Face login; both model and dataset repositories are private.
Original project-native data files are available under `source/` in the dataset repository.
Canonical dataset fingerprint: `d41f8adb26fb5c10e4f7a0747b5e351c4755d98afc4199cc48fe53aef54bb7d6`.
