---
library_name: transformers
pipeline_tag: text-generation
tags:
- advai-notebooks
- llm-base
- demo
- synthetic-data
- random-initialization
- full
- llama
language:
- en
datasets:
- widebluesky/wbs-llm-base-demo
---

# widebluesky/wbs-llm-base-demo

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
| Training objective | next-token causal language modeling over document text (continued pre-training) |
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

Training supervises the next token in each document, including its EOS marker.
Padding tokens are masked from the loss. A chat template is included as tokenizer
metadata for compatibility; the Base training objective does not teach chat behavior.

## Synthetic held-out test metrics

| Metric | Before training | Selected trained export |
| --- | ---: | ---: |
| nll | 4.792373 | 4.695202 |
| perplexity | 120.587147 | 109.420896 |

NLL and perplexity are computed over each task's supervised tokens.
Base and Instruct perplexity values are not directly comparable because their
supervision ranges differ. These results use only two synthetic test examples.

## Inference

In a checkout of `advai-notebooks`:

```bash
python -m pip install -e '.[llm-base]'
```

```python
from advai_notebooks.llm.base.model import BaseModel

model = BaseModel.load("widebluesky/wbs-llm-base-demo", revision="demo-v1", device="cpu")
outputs = model.generate(["the sky is"], max_new_tokens=8)
print(outputs)
```

The full model also supports native Hugging Face loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "widebluesky/wbs-llm-base-demo"
model = AutoModelForCausalLM.from_pretrained(repo_id, revision="demo-v1")
tokenizer = AutoTokenizer.from_pretrained(repo_id, revision="demo-v1")
```

For Instruct inference, render the exported chat template with an assistant generation
For Base inference, tokenize the document prompt and generate a continuation.
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

Dataset: [widebluesky/wbs-llm-base-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-base-demo).
Verified dataset revision: [`de84c42b70656ff5875602b1abf829c8b2c962f9`](https://huggingface.co/datasets/widebluesky/wbs-llm-base-demo/tree/de84c42b70656ff5875602b1abf829c8b2c962f9) (`demo-v1`).

This is the synthetic source dataset used for this trained demo release.
Its record fingerprint matches the saved training-run provenance; the published
dataset was reloaded and checked record-by-record before associating it with this model.

The `base` configuration contains 8 train, 2 validation and 2 test records.
Original document/conversation group IDs and split assignments are preserved.

```python
from datasets import load_dataset

data = load_dataset(
    "widebluesky/wbs-llm-base-demo", "base",
    revision="de84c42b70656ff5875602b1abf829c8b2c962f9", token=True,
)
```

Access uses the existing Hugging Face login; both model and dataset repositories are private.
Original project-native data files are available under `source/` in the dataset repository.
Canonical dataset fingerprint: `26aec5cfd5b21082af1e8f772dfe41f447ca0275f5cc6b8b8b1d98334e791335`.
