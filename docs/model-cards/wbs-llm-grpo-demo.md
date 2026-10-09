---
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- advai-notebooks
- llm-grpo
- grpo
- reinforcement-learning
- demo
- synthetic-data
- full
- llama
base_model:
- widebluesky/wbs-llm-instruct-demo
base_model_relation: finetune
datasets:
- widebluesky/wbs-llm-grpo-demo
---

# widebluesky/wbs-llm-grpo-demo

## Demo release v1

A **trained tiny GRPO demo model** for verifying the training, export, publication,
download and inference lifecycle. This is a full-parameter online RL fine-tune of
[the tiny Instruct SFT demo](https://huggingface.co/widebluesky/wbs-llm-instruct-demo/tree/011592e1d331b07c7948272fe8a512adb297cf19),
fixed initial revision `011592e1d331b07c7948272fe8a512adb297cf19`.
The initial model was a randomly initialized tiny Llama architecture trained on
synthetic data; it does not contain Meta Llama pretrained weights.
This model is a workflow placeholder, with no demonstrated general language capability.

## Model and training

Group relative policy optimization with terminal rule rewards, group-normalized advantages, a clipped policy loss and reference KL regularization; no separate critic.

| Setting | Value |
| --- | --- |
| Architecture / actor parameters | LlamaForCausalLM / 13,248 |
| Layers / hidden size / FFN size | 1 / 32 / 64 |
| Attention heads / key-value heads | 4 / 2 |
| Vocabulary / native context limit | 123 / 128 tokens |
| Configured prompt + generation limit | 64 tokens; over-budget inputs rejected |
| Training method | Full parameters; standalone model, not an adapter |
| Epochs completed / selected | 2 / 2 |
| Optimizer updates / sampled completions | 32 / 32 |
| Prompt batch size / update epochs / minibatch size | 2 / 2 / 2 |
| Optimizer / learning rate / weight decay | AdamW / 0.0005 / 0.0 |
| Policy clip range / KL coefficient | 0.2 / 0.05 |
| Sampling temperature / max completion tokens | 1.0 / 8 |
| Seed / device / precision | 42 / CPU / fp32 |
| Selection | Maximum validation mean reward, including epoch zero; later epoch wins ties |
| Held-out generation | Greedy; test is report-only |

Four sampled completions per prompt; group standard deviation uses population variance and a 1e-6 floor. Constant-reward groups have zero reward advantages. Reference KL uses the nonnegative k3 estimator.

The terminal reward combines exact match (weight 1.0), whitespace-token overlap F1
(weight 1.0), and length closeness (weight 0.2; target 4 words). JSON validity and
custom rewards are disabled. Reference answers are available only to reward and
evaluation code; they are not appended to policy prompts.
PPO and GRPO use separate synthetic datasets with matching question/answer content.

## Training and evaluation dataset

Dataset: [widebluesky/wbs-llm-grpo-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-grpo-demo).
Fixed verified revision: [`3246400626f9c419fd1eaa4b36a1594b53089be2`](https://huggingface.co/datasets/widebluesky/wbs-llm-grpo-demo/tree/3246400626f9c419fd1eaa4b36a1594b53089be2) (`demo-v1`).
The `grpo` configuration contains train 4, validation 2 and test 2 records.
Source bytes and the canonical training-data fingerprint match the saved training run.
The private Hub dataset was reloaded and compared record by record before association.

```python
from datasets import load_dataset

data = load_dataset(
    "widebluesky/wbs-llm-grpo-demo", "grpo",
    revision="3246400626f9c419fd1eaa4b36a1594b53089be2", token=True,
)
```

## Synthetic held-out results and limitations

| Metric | Before RL | Selected export |
| --- | ---: | ---: |
| Mean reward | 0.0 | 0.0 |
| Exact match | 0.0 | 0.0 |
| Mean completion tokens | 8.0 | 8.0 |

Only two synthetic test records were evaluated. Both before and after RL, exact
match is 0/2. Generation may be repetitive or incorrect. Parameter updates and
successful lifecycle checks do **not** demonstrate improved answer quality.
This release is unsuitable for production use or meaningful comparisons of RL methods.

## Inference

Install this project from a checkout:

```bash
python -m pip install -e '.[llm-grpo]'
```

```python
from advai_notebooks.llm.grpo.model import GRPOModel

model = GRPOModel.load("widebluesky/wbs-llm-grpo-demo", revision="demo-v1", device="cpu")
response = model.chat(
    [{"role": "user", "content": "what color is the sky ?"}],
    max_new_tokens=8,
)
print(response)
```

The complete actor can also be loaded natively:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "widebluesky/wbs-llm-grpo-demo"
model = AutoModelForCausalLM.from_pretrained(repo_id, revision="demo-v1")
tokenizer = AutoTokenizer.from_pretrained(repo_id, revision="demo-v1")
```

Render the exported chat template with an assistant generation prompt for native
chat inference. Pin the model publication commit SHA for reproducible deployment.
Private repository access uses your existing Hugging Face login.
The full actor does not need the initial SFT repository at inference time.

## Provenance and verification

Training run: `20261007-6dfcf34d`. Source implementation commit:
`370b95d9448f8914871f01a268fca028640e12e8`.
Initial SFT actor SHA256: `5dba90066e634e5e03327a0881969749afbe61f709686bc2080d38808f4a6339`.
Published actor SHA256: `f2341a67da544accd9408439219bf270dcbe0a844db7f02db096728bf38df320`.

The actor tensors changed from the initial SFT policy. The training reference stayed
frozen, and exported weights match the validation-selected epoch 2 checkpoint.
Local fresh-process reload reproduced held-out predictions and metrics exactly.
Native Transformers and the project facade produced equal logits; CLI and API
generate/chat outputs matched. Publication packages are loaded before upload.

Inference weights, architecture and generation configuration, tokenizer, chat
template, facade metadata, this English card, and a SHA256 manifest are included.
Raw data, frozen reference checkpoints, optimizer/RNG state, training checkpoints,
logs and credentials are excluded.
