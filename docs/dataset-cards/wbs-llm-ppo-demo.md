---
language:
- en
task_categories:
- text-generation
size_categories:
- n<1K
tags:
- advai-notebooks
- ppo
- reinforcement-learning
- synthetic
- demo
configs:
- config_name: ppo
  default: true
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# widebluesky/wbs-llm-ppo-demo

Independent synthetic prompt/reference dataset used to train the PPO demo model.
This repository contains only this task's data and can be loaded independently.

## Contents and split policy

Eight English fixtures: train 4, validation 2, test 2. Fields are `id`, `group_id`,
`split`, `messages` (role/content pairs ending in a user turn), and `reference`.
The reference is used to compute reward and evaluation; it is never appended to policy inputs.
Group IDs and prompt content are disjoint across splits. The two RL demo datasets
have the same question/answer content, with separate task IDs and versioned repositories.
These small splits are engineering fixtures, not a meaningful generalization benchmark.

`source/ppo.jsonl` preserves the training input byte for byte. `data/` contains split
views. `dataset_manifest.json` records source hashes, row fingerprints, the training
run, reward configuration, and initial SFT model provenance.

## Loading

```python
from datasets import load_dataset

data = load_dataset("widebluesky/wbs-llm-ppo-demo", "ppo", revision="demo-v1", token=True)
```

Pin the dataset commit SHA for reproducibility. The repository is private; `token=True`
uses your existing Hugging Face login. Validation uses Datasets 4.8.5.
For this project's training CLI, download the dataset snapshot and set
`data.samples` to `source/ppo.jsonl`.

## Training association and limitations

Used by [widebluesky/wbs-llm-ppo-demo](https://huggingface.co/widebluesky/wbs-llm-ppo-demo) in training run `20261007-6dfcf34d`.
The initial policy is the tiny SFT demo, not a pretrained production language model.
Training uses rule-based exact-match, token-overlap and length rewards.
The selected model's held-out exact match and mean reward are both 0.0.
Data are synthetic, contain no real user conversations, and support lifecycle verification
only. No claim of production readiness, quality improvement, or representative coverage.
