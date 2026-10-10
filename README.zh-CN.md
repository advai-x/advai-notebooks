<p align="center">
  <img src="docs/assets/readme-hero.png" alt="advai-notebooks — 基于 Notebook 的实用、可复现模型训练" width="100%">
</p>

<h1 align="center">advai-notebooks</h1>

<p align="center">
  <a href="README.md">English</a> | 简体中文 | <a href="https://advai-x.github.io/advai-notebooks/zh-cn/">文档站</a>
</p>

<p align="center">
  面向现代 AI 模型的实用、可复现训练指南。<br>
  从离线 tiny smoke 实验，到使用自己数据的可配置训练。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="许可证：MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-training-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 训练"></a>
  <a href="notebooks/"><img src="https://img.shields.io/badge/Google%20Colab-ready-F9AB00?logo=googlecolab&amp;logoColor=white" alt="支持 Google Colab"></a>
  <a href="docs/publishing.zh-CN.md"><img src="https://img.shields.io/badge/Hugging%20Face-export%20ready-FFD21E" alt="支持 Hugging Face 导出"></a>
</p>

advai-notebooks 提供可执行 Notebook、YAML 配置和可复用 Python 模块，帮助学习和实践主流模型的训练流程。
每条已实现路径都覆盖完整生命周期：

```text
校验数据 → 配置模型 → 训练 → 评估 → 断点恢复 → 导出 → 重新加载
```

Notebook 以教学和编排为主，调用与 CLI 相同、经过测试的实现。可以在 Google Colab 交互学习，
再将相同配置迁移到本地或 GPU 环境。默认文档使用英文；本页、参考指南和 Notebook 提供中文版。
目录级 README、模型卡和数据卡目前以英文为主。

> [!IMPORTANT]
> 仓库中的 smoke 配置使用随机 tiny 模型和合成数据，仅验证端到端工程链路。
> 示例指标与生成内容不代表真实模型质量。

## 选择训练指南

| 路径 | 训练内容 | 支持方法 | 中文入口 |
| --- | --- | --- | --- |
| Embedding | 文本稠密检索编码器 | MNRL、Cached MNRL、困难负例准备 | [Notebook](notebooks/embedding/01_finetune.zh-CN.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/embedding/01_finetune.zh-CN.ipynb) |
| Ranker | CrossEncoder 重排模型 | Pointwise BCE、排序评估 | [Notebook](notebooks/ranker/01_finetune.zh-CN.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/ranker/01_finetune.zh-CN.ipynb) |
| Decision | choice、score、yes/no 类型化决策 | CE、有序损失、可选 RLCD、校准 | [Notebook](notebooks/decision/01_finetune.zh-CN.ipynb) · [架构指南](docs/architecture.zh-CN.md) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/decision/01_finetune.zh-CN.ipynb) |
| LLM Base | Decoder 语言模型 | 继续预训练、full、LoRA | [Notebook](notebooks/llm/base/01_finetune.zh-CN.ipynb) · [指南](docs/llm.zh-CN.md) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/base/01_finetune.zh-CN.ipynb) |
| LLM Instruct | 指令助手 | Assistant-only SFT、full、LoRA | [Notebook](notebooks/llm/instruct/01_finetune.zh-CN.ipynb) · [指南](docs/llm.zh-CN.md) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/instruct/01_finetune.zh-CN.ipynb) |
| LLM PPO | 带价值头的语言策略 | GAE、clipped policy/value、full/LoRA | [RL 指南](docs/rl.zh-CN.md) · [配置](configs/llm/ppo/) |
| LLM GRPO | 无 critic 的语言策略 | 组内 advantage、clipping、reference KL、full/LoRA | [RL 指南](docs/rl.zh-CN.md) · [配置](configs/llm/grpo/) |
| Diffusion | Latent 文生图模型 | Full UNet、attention LoRA | [Notebook](notebooks/diffusion/01_finetune.zh-CN.ipynb) · [指南](docs/diffusion.zh-CN.md) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/diffusion/01_finetune.zh-CN.ipynb) |
| VLM | 原生 LLaVA 单图问答 | 回答区监督 full / 语言 LoRA | [Notebook](notebooks/vlm/01_finetune.zh-CN.ipynb) · [指南](docs/vlm.zh-CN.md) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vlm/01_finetune.zh-CN.ipynb) |
| VLA | CLIP + 语言 + 状态 → 连续动作 chunk | 行为克隆基线，冻结或更新编码器 | [Notebook](notebooks/vla/01_finetune.zh-CN.ipynb) · [指南](docs/vla.zh-CN.md) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vla/01_finetune.zh-CN.ipynb) |
| VM | 动作条件数值 World Model | Residual dynamics、原生权重微调、递归 rollout | [Notebook](notebooks/vm/01_finetune.zh-CN.ipynb) · [指南](docs/vm.zh-CN.md) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vm/01_finetune.zh-CN.ipynb) |

VLM、VLA、VM 与 `llm/` 平级；`vm/` 明确表示 World Model，而非 Vision/Video Model。
当前后端范围有限：VLM 不实现视频/Qwen-VL，VLA 不加载 OpenVLA checkpoint，VM 不生成视频。
VLA/VM 是可训练基线，并非已验证的机器人控制或规划能力。

另有可选的 [检索组合对照 Notebook](notebooks/workflows/01_retrieval_comparison.zh-CN.ipynb)，
同时评估 embedding 召回和 ranker 重排；两类模型仍可独立训练。

## 快速开始

### 方式 A：Google Colab

从[训练目录](notebooks/README.zh-CN.md)打开 Notebook，在 Colab runtime 中准备仓库：

```python
%cd /content
!test -d advai-notebooks || git clone --depth 1 https://github.com/advai-x/advai-notebooks.git
%cd /content/advai-notebooks
%pip install -e ".[embedding,notebooks]"
```

将 `embedding` 替换为所选任务的 extra：`ranker`、`decision`、`llm-base`、`llm-instruct`、
`llm-ppo`、`llm-grpo`、`diffusion`、`vlm`、`vla` 或 `vm`。
正式训练的数据与 checkpoint 应放入 Google Drive 等持久化存储；runtime 断开后，
仅保存在 `/content` 的文件可能丢失。

### 方式 B：本地 smoke

```bash
git clone https://github.com/advai-x/advai-notebooks.git
cd advai-notebooks
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[embedding,notebooks,dev]'
python -m advai_notebooks.embedding validate --config configs/embedding/smoke.yaml
python -m advai_notebooks.embedding run --config configs/embedding/smoke.yaml
```

依赖安装完成后 smoke 可以离线运行；产物写入被 Git 忽略的 `outputs/`。

## 从 smoke 到真实实验

1. 选择最接近的指南，先运行未经修改的 smoke 配置。
2. 复制对应的 `hf.yaml`、`bge.yaml` 或 `lora.yaml`。
3. 固定模型 ID/revision，并指向自己按来源分组的数据 split。
4. 先校验 smoke。真实配置的模型感知校验会加载权重，可能需要目标硬件和足够内存。
5. 使用新的输出目录，保存实验配置。
6. 在 held-out 数据上评估并检查报告，再显式导出或发布。

```bash
cp configs/llm/instruct/hf.yaml configs/llm/instruct/my_experiment.yaml
python -m advai_notebooks.llm.instruct validate --config configs/llm/instruct/my_experiment.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/my_experiment.yaml
```

多数训练器支持 epoch 边界恢复：

```bash
python -m advai_notebooks.embedding run \
  --config configs/embedding/my_experiment.yaml \
  --resume outputs/embedding/my_experiment/latest
```

恢复要求原配置、数据、相关源码指纹和包版本保持一致；更改实验条件应使用新输出目录。

## 项目特点

- 完整生命周期：校验、训练、评估、checkpoint、导出和重新加载。
- Notebook/CLI 使用相同模块，教学实验可以迁移到自动化运行。
- 随机 tiny 模型与合成 fixtures 优先检查工程问题，不替代真实质量评估。
- YAML 配置管理模型来源、数据路径、运行参数和超参数。
- 各任务独立维护自己的数据、损失和指标；应用组合显式表达。
- 按分组、split 和标签契约检查数据泄漏；语义近重复仍需人工处理。
- 保存配置、环境、指标、checkpoint 和模型元数据。
- 发布支持离线 dry-run，训练不自动上传原始数据。

## CLI 与可选流程

所有任务遵循相同的基本命令形态：

```text
python -m advai_notebooks.<task> validate --config <config.yaml>
python -m advai_notebooks.<task> run --config <config.yaml>
```

`<task>` 可为 `embedding`、`ranker`、`decision`、`llm.base`、`llm.instruct`、`llm.ppo`、
`llm.grpo`、`diffusion`、`vlm`、`vla` 或 `vm`，对应配置在 `configs/` 同名目录。
数据准备与检索组合评估是独立、可选阶段：

```bash
python -m advai_notebooks.data_preparation prepare --config configs/data_preparation/smoke.yaml
python -m advai_notebooks.workflows compare \
  --config configs/workflows/smoke.yaml \
  --embedding-run outputs/embedding/smoke \
  --ranker-run outputs/ranker/smoke \
  --output outputs/retrieval/comparison.json
```

## 仓库结构

```text
advai-notebooks/
├── notebooks/                  # 中英双语可执行教程
│   ├── embedding/ ranker/ decision/
│   ├── llm/{base,instruct}/
│   ├── vlm/ vla/ vm/ diffusion/
│   └── workflows/
├── configs/                    # Smoke、真实基座和 LoRA 配置
├── src/advai_notebooks/         # 可复用实现：任务目录 + common
├── data/sample/                # 可版本化的小型样例与 schema
├── docs/                       # 架构、任务、数据和发布参考指南
├── site/                       # 双语文档站与内容转换脚本
├── tests/                      # 单元、契约和生命周期回归测试
└── outputs/                    # 本地生成产物，Git 忽略
```

## 中文参考文档

[文档站](https://advai-x.github.io/advai-notebooks/zh-cn/)提供搜索、分类导航与对应语言切换。
启用 GitHub Pages 并部署文档工作流后，站点才会可用。本地预览与设置步骤见[文档站指南](site/README.zh-CN.md)。
以下原始文件仍是指南正文的唯一来源。

- [架构与扩展边界](docs/architecture.zh-CN.md)
- [Demo 数据与 Hugging Face Dataset](docs/datasets.zh-CN.md)
- [LLM Base / Instruct](docs/llm.zh-CN.md)
- [PPO / GRPO](docs/rl.zh-CN.md)
- [VLM](docs/vlm.zh-CN.md)、[VLA](docs/vla.zh-CN.md)、[VM / World Model](docs/vm.zh-CN.md)
- [Diffusion](docs/diffusion.zh-CN.md)
- [Hugging Face 发布](docs/publishing.zh-CN.md)
- [Notebook 目录](notebooks/README.zh-CN.md)
- [检索数据契约（英文）](data/sample/retrieval/README.md)

## 数据、凭据与生成产物

大型或敏感数据不要提交到 Git；`data/local/` 已忽略。Token 放入环境变量或 Colab Secrets，
不要写进 Notebook/YAML。`outputs/` 的模型权重、checkpoint 和生成结果不提交。
小型样例、schema 和 manifest 也仅在允许再分发时提交。下载、训练或发布前检查模型与数据许可证。

## 开发与贡献

安装所修改任务的 extra 和 `dev`，例如：

```bash
python -m pip install -e '.[embedding,dev]'
python -m pytest tests/embedding
python -m ruff check src tests
```

完整测试依赖多个可选 extra。贡献者可先运行任务测试，明确记录真实 checkpoint 的硬件验证范围。
欢迎 issue、纠错和新增训练指南：说明模型族、训练目标和预期输出；将可复用逻辑放进 `src/advai_notebooks/`，
Notebook 保持教学用途；尽量添加 tiny fixtures/smoke 配置，以及数据、指标、恢复、导出测试。
更新硬件限制和已知边界，不提交私有数据、凭据或大型权重。

英文为默认文档语言，中文版本使用 `*.zh-CN.md` / `*.zh-CN.ipynb` 并提供切换链接。
数据契约变更时同步两版。模型要求的语言前缀与真实数据内容不随文档翻译改变。

## 状态与许可证

当前版本为 `0.1.0`，处于活跃开发阶段。Smoke 以可复现为目标；真实模型配置是起点，
必须在自己的数据、硬件和质量标准下验证。示例不代表多 GPU 或生产规模性能支持。

除另有说明外，本仓库的原创代码、Notebook、配置和文档均采用 [MIT 许可证](LICENSE)。
复制或再分发软件及其重要部分时，请保留该文件中的版权声明和许可声明。

第三方依赖、模型权重和数据集仍遵循各自的许可证与使用条款。
使用本项目训练，不会自动使生成的 checkpoint 或 adapter 获得 MIT 授权；
使用或发布前，请检查适用的授权条件与限制。
