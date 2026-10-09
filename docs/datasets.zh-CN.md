# Demo 模型与 Dataset 关联

[English](datasets.md) | 简体中文

> 本页记录既有 demo 发布与验证历史，不表示已重新检查当前 Hub 的可见性、分支或标签状态。

目前已发布的七个模型分别对应七份私有 Hugging Face Dataset，模型卡与数据卡均使用英文。
Embedding 与 Ranker 的 Dataset 仓库、训练样本和版本分别维护。
两份包各自包含完整的评估基准副本，加载时不依赖另一个 Dataset；当前 demo 基准内容相同。

| 模型 | Dataset | 训练配置 |
|---|---|---|
| `widebluesky/wbs-embedding-demo` | [widebluesky/wbs-embedding-demo](https://huggingface.co/datasets/widebluesky/wbs-embedding-demo) | `embedding_training` |
| `widebluesky/wbs-ranker-demo` | [widebluesky/wbs-ranker-demo](https://huggingface.co/datasets/widebluesky/wbs-ranker-demo) | `ranker_training` |
| `widebluesky/wbs-decision-demo` | [widebluesky/wbs-decision-demo](https://huggingface.co/datasets/widebluesky/wbs-decision-demo) | `triage` |
| `widebluesky/wbs-llm-base-demo` | [widebluesky/wbs-llm-base-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-base-demo) | `base` |
| `widebluesky/wbs-llm-instruct-demo` | [widebluesky/wbs-llm-instruct-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-instruct-demo) | `instruct` |
| `widebluesky/wbs-llm-ppo-demo` | [widebluesky/wbs-llm-ppo-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-ppo-demo) | `ppo` |
| `widebluesky/wbs-llm-grpo-demo` | [widebluesky/wbs-llm-grpo-demo](https://huggingface.co/datasets/widebluesky/wbs-llm-grpo-demo) | `grpo` |

## 数据版本

各 Dataset 的 `demo-v1` 标签固定到下列已验证提交；模型卡正文也记录这些 SHA。

| Dataset | 固定提交 |
|---|---|
| `widebluesky/wbs-embedding-demo` | `d5d422450670f840377c9f8bb3b11b7919cdadfa` |
| `widebluesky/wbs-ranker-demo` | `356d77b9736aa8185480751a9e849d60cfebb6e5` |
| `widebluesky/wbs-decision-demo` | `ccdf91daf6b2810dbf63eda37e69dd2ab3af8f6c` |
| `widebluesky/wbs-llm-base-demo` | `de84c42b70656ff5875602b1abf829c8b2c962f9` |
| `widebluesky/wbs-llm-instruct-demo` | `21213b40998690495308dac1dae09d669846579b` |
| `widebluesky/wbs-llm-ppo-demo` | `5b621b1b02f8f097a2036c0aa0f915e4df5b8e31` |
| `widebluesky/wbs-llm-grpo-demo` | `3246400626f9c419fd1eaa4b36a1594b53089be2` |

模型主分支的英文模型卡通过标准 `datasets` 元数据字段关联上述仓库，并提供固定数据版本和加载示例。
此前五个模型的 Dataset 关联更新只修改模型卡和 README 校验记录，原有模型 `demo-v1` 标签保持原提交。
PPO/GRPO 本次发布训练后的完整模型及各自 Dataset，并在模型卡关联固定数据 SHA。
Dataset 和模型可以使用相同 repo ID，但 repo_type 不同；
Dataset 页面 URL 必须包含 `/datasets/`。

## 加载与项目训练输入

在独立数据检查环境中安装已验证版本：

```bash
python -m pip install 'datasets==4.8.5'
```

```python
from datasets import load_dataset

data = load_dataset(
    "widebluesky/wbs-llm-instruct-demo", "instruct",
    revision="21213b40998690495308dac1dae09d669846579b", token=True,
)
print(data)
```

`token=True` 使用已有 Hugging Face 登录，不应在代码中写入 Token。全部仓库保持私有。
已检查从 Hub 固定 SHA 直接调用 `load_dataset`，不仅是从本地文件重载。

每份 Dataset 在 `source/` 保留原始文件；`data/` 为按 split 导出的 JSONL 视图。
需要使用现有项目训练器时，下载 Dataset snapshot，将任务配置的数据路径指向 `source/` 原文件：

- Embedding：`corpus.jsonl`、`queries.jsonl`、`qrels.jsonl`、`embedding_samples.json`。
- Ranker：`corpus.jsonl`、`queries.jsonl`、`qrels.jsonl`、`ranker_samples.json`、`candidates.json`。
- Decision：`triage.jsonl` 和 `tasks.json`。
- Base：`base.jsonl`。
- Instruct：`instruct.jsonl`。
- PPO：`ppo.jsonl`。
- GRPO：`grpo.jsonl`。

Embedding 有 4 个配置：`queries`、`corpus`、`qrels`、`embedding_training`（默认配置，train 24 条）。
Ranker 有 5 个配置：`queries`、`corpus`、`qrels`、`ranker_training`（默认配置，train 48 条）、`candidates`。
`queries` 保留 train 12 / validation 3 / test 3；`corpus` 有 12 篇文档，`qrels` 有 216 条判断。
训练样本配置只有 train，评估 split 在独立的 benchmark 配置中。
Decision 的 `triage` 保留独立 calibration split；Base/Instruct 为各自 train/validation/test。
PPO/GRPO 各只有一个默认配置，分别包含 train 4 / validation 2 / test 2；reference 仅用于奖励和评估。

## 验证范围

- 数据格式、唯一 ID、分组和重复内容跨 split 检查、引用及正负标签规则。
- 数据指纹与产生已发布模型权重的训练记录匹配。
- 当前 7 份 Dataset 共 14 个配置。PPO/GRPO 新发布的 2 份、2 个配置在本地和 Hub 均逐条重载通过，默认配置也验证通过；原有 5 份沿用已验证的固定版本。
- 每份拆分包仅包含本任务训练样本，并能独立通过项目数据校验与标准 `load_dataset` 加载。
- 原始文件字节及发布文件 SHA256 一致。
- 模型卡的 Dataset 元数据、固定数据版本与 README 清单一致；其余模型文件哈希保持一致。

数据均为英文合成 smoke fixtures，验证工程链路与来源一致性，不构成业务质量或泛化效果评估。

完整来源与配置指纹保存在各 Dataset 的 `dataset_manifest.json`；数据卡源文件位于 [dataset-cards](dataset-cards/)。
后续发布真实模型前，应先验证并发布相应 Dataset，再在模型卡中关联 Dataset ID 和固定提交。
本目录的 demo 卡片仅适用于这些已验证数据与权重，不应直接复用到使用其他语料训练的版本。

## 共享检索 Dataset 的迁移

原 `widebluesky/wbs-retrieval-demo` 保留为历史组合数据集，原 `demo-v1` 固定提交为
`c75ca4e872c63c4683b266a4924a493968a1111f`。当前 Embedding/Ranker 模型主分支仅关联各自的 Dataset。
旧数据和模型 `demo-v1` 标签保持原提交，便于复现历史发布。以后两份任务 Dataset 可以分别调整、发布。
本次迁移仅拆分 Dataset 包和更新模型卡关联，已发布模型权重保持一致。

PPO/GRPO 的模型和独立 Dataset 均已发布为私有 `demo-v1`，并已验证 Hub 下载后的文件哈希、推理输出和测试报告。
固定模型/数据 SHA 及验证记录见 [RL demo 发布记录](releases/20261008-rl-demo-v1.json)。
两份模型的测试 exact match 和 mean reward 均为 0.0，仅用于验证完整工程链路。
