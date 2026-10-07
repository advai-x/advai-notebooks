# LLM Base 与 Instruct 微调

两条独立路径分别实现领域继续预训练（Base）与指令监督微调（Instruct SFT）。
各自拥有配置、数据、采样、损失、优化器、训练、评估、指标、模型门面、CLI 和发布模块。
底层复用 Transformers 的原生 decoder/LM head 和 Trainer；LoRA 使用 PEFT。
不会导入另一个 LLM 任务或 Embedding/Ranker/Decision。

## 安装与运行

```bash
python -m pip install -e '.[llm-base,llm-instruct,notebooks,dev]'
python -m advai_notebooks.llm.base validate --config configs/llm/base/smoke.yaml
python -m advai_notebooks.llm.base run --config configs/llm/base/smoke.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/smoke.yaml
```

也可使用 `advai-llm-base`、`advai-llm-instruct`。每个任务有 `smoke.yaml`、`lora.yaml`、
`hf.yaml`。`hf.yaml` 是真实模型配置示例，必须换成自己的数据；未在此进行领域训练或质量评估。
单独安装任一任务 extra 足以运行该任务。已验证的 SDK 版本是 Transformers 5.18.0、
PEFT 0.18.1、Accelerate 1.15.0。

## 数据契约

Base JSONL：

```json
{"id":"doc-1","document_id":"book-1","text":"The sky is blue.","split":"train"}
```

Instruct JSONL：

```json
{"id":"conversation-1","conversation_id":"session-1","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What color is the sky?"},{"role":"assistant","content":"Blue."}],"split":"train"}
```

也支持用 `prompt`、`completion` 替代 `messages`，归一化为 user/assistant。
第一版是纯文本对话，可选首条 system，随后 user/assistant 交替；训练样本以 assistant 结束，
推理请求以 user 结束。工具调用、多模态消息暂未实现。

三个 split 为 `train`、`validation`、`test`，都不能为空。按 document/conversation 分组划分后
再编码、切片；禁止重复 ID、跨 split 的同组或相同内容。来源关系需要调用方用分组 ID 表达，
当前校验不会识别语义改写后的重复数据。

Base 在每份文档末尾加入 EOS。`reject_truncation: false` 按长度切片，相邻片段共享一个上下文
token，保证边界监督 token 不遗漏、不重复，不将不同文档拼接。设为 true 则拒绝超长文档。
Padding 按位置屏蔽，即使 pad 与 EOS 使用相同 ID，也保留真实 EOS 的监督。

Instruct 只监督 assistant 区域。优先使用模板 `{% generation %}` 区间与 fast tokenizer offsets，
不会把尾部空白映射为后续用户消息。没有 generation 区块时，验证渲染前缀稳定后计算回复区域；
前缀不稳定、token 跨越边界时明确拒绝，需提供 `model.chat_template` Jinja 文本。
模板必须与基座词表、训练目标及推理格式相匹配，不能随意替换角色标记。
`reject_truncation: true` 默认拒绝超长会话；显式 false 时从右侧截断，统计丢弃 token，
且至少保留一个可监督回复 token。应先整理领域会话长度，避免截断末尾答案或结束标记。

## 训练、评估与恢复

`training.method` 为 `full` 或 `lora`。可配置 optimizer、学习率、weight decay、epoch、batch、
梯度累积、梯度裁剪、gradient checkpointing、seed，以及 LoRA rank/alpha/dropout/target modules。
第一版使用 constant 学习率，单设备 Trainer；不自动选择多卡、量化或模型切分。
CPU/MPS 使用 fp32，CUDA 可配置 fp16/bf16，需目标设备支持。

生命周期：验证数据与模板 → baseline validation/test → 训练 → 每 epoch validation →
按 validation NLL 选模（含 epoch 0）→ test 对比 → 导出。Test 不影响训练或选模。
NLL 按有效、shift 后的监督 token 加权，PPL=exp(NLL)，不是 batch 平均 loss 的平均值。
不同 tokenizer、语料或监督范围下的 PPL 不宜直接比较。
两任务的 `metrics.py` 独立维护公式、指标记录与选模。Instruct 额外可配置
`generation_metrics: [exact_match, json_validity]`，按会话计数；生成评估固定 greedy，
与推理采样配置独立。低 NLL 不等于具备业务指令能力。

```text
output_dir/
  config.json, environment.json, data_summary.json
  baseline/                         # 含 epoch-zero 模型
  baseline_validation.json, baseline_test.json
  trainer/checkpoint-N/             # native Trainer 模型 + optimizer/scheduler/RNG
    advai_metadata.json             # 推理门面信息
    advai_resume.json               # 签名、验证历史、最佳 checkpoint 指针
  latest.json                       # 最近完整 checkpoint
  best/                             # validation 选中的推理模型
  export/                           # 用于部署/发布的推理模型
  report.json
```

```bash
python -m advai_notebooks.llm.base run --config configs/llm/base/smoke.yaml \
  --resume outputs/llm/base/run/latest
```

`latest` 为指向 `latest.json` 的逻辑别名，也可指定完整 `trainer/checkpoint-N`。
推理 export 不含 optimizer/scheduler/RNG，不能续训。恢复要求计划 epoch、配置、数据、
包版本、设备、tokenizer/template 以及相关源码指纹保持一致；调整实验应选择新的输出目录。
完整 checkpoint 以 `advai_resume.json` 完成标志为准；未完成 epoch 从上一个保存边界重放。
CPU/fp32 已测试 full/LoRA 的中断与连续训练权重一致；CUDA/MPS/AMP 恢复须在目标硬件验证。

## 模型、推理与衔接

```python
from advai_notebooks.llm.base.model import BaseModel
from advai_notebooks.llm.instruct.model import InstructModel

base = BaseModel.load("outputs/llm/base/run/export")
texts = base.generate(["The sky is"], max_new_tokens=8)

instruct = InstructModel.load("outputs/llm/instruct/run/export")
answer = instruct.chat([{"role": "user", "content": "What color is the sky?"}], max_new_tokens=8)
```

`generate` 返回与输入同序的续写列表，`chat` 返回一个回复字符串，都不包含输入 prompt。
默认 greedy，可配置 `do_sample`、`temperature`、`top_p` 和 `max_new_tokens`。
CLI `generate --input` 接收字符串列表，`chat --input` 接收会话列表（每个会话为 messages 数组）。
两者将 `{"outputs": [...]}` 写入 `--output` JSON。
Prompt 加生成预算不能超过 `model.max_len`，不会静默截断。模型原生 generation config
会随完整模型保存，门面配置通过 `advai_metadata.json` 保存。

Base→Instruct 是可选的产物衔接：将 Instruct `model.id` 设为 Base 的 full/merged 导出目录，
同时确认或提供适配的 chat template。训练初始化不接受 adapter-only 目录，应先合并。
通过原生文件契约加载，不调用 Base 任务代码。

## Full、adapter 与 merged 导出

Full 保存原生权重/config、tokenizer/template、generation config 与门面元信息，支持
`AutoModelForCausalLM.from_pretrained`。LoRA 保存 adapter 和 tokenizer/template，记录基座来源。
Hub 基座固定到解析出的 commit SHA；tiny/本地基座附带冻结的 `base_model/` 快照，方便搬迁。
因此本地基座 adapter 包可能很大，配置时需考虑存储成本。加载 bundled adapter 推荐模型门面；
原生 PEFT 需先从该目录的 `base_model/` 显式加载基座。

```bash
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/lora.yaml
python -m advai_notebooks.llm.instruct export \
  --checkpoint outputs/llm/instruct/lora-run/export \
  --output outputs/llm/instruct/merged-export --merged
```

合并通过重载副本执行 safe merge，不修改活动模型；merged 可独立加载，不需要 PEFT adapter。
第一版未支持量化 adapter，不承诺 QLoRA 合并行为。

## Hugging Face 发布

```bash
python -m advai_notebooks.llm.base publish \
  --export-dir outputs/llm/base/run/export \
  --repo-id widebluesky/wbs-llm-base-demo --dry-run \
  --output outputs/publishing/llm-base-plan.json

python -m advai_notebooks.llm.instruct publish \
  --export-dir outputs/llm/instruct/run/export \
  --repo-id widebluesky/wbs-llm-instruct-demo --dry-run \
  --output outputs/publishing/llm-instruct-plan.json
```

发布默认 private，去掉 `--dry-run` 才上传；训练和 Notebook 默认不会上传。
发布包只包含原生推理资产、英文模型卡与 SHA256 清单，排除原始会话、训练状态、日志和凭据。
Adapter 的 tokenizer、chat template、固定基座契约都进入发布包；有 bundled base 时一并上传。
离线预览强制使用缓存验证基座，未缓存的远程基座会报错而不会下载。
Hub 加载支持 `revision`、`cache_dir`、`local_files_only`，部署建议固定发布 commit SHA。
本次实现未创建或上传新的 LLM 仓库。

## 当前范围

已实现 Base CPT、Instruct SFT、full/LoRA、单设备、原生导出/重载、验证选模、epoch 恢复、
CLI、Notebook 和发布入口。离线 tiny 数据只用于功能验证。
QLoRA、跨文档 packing、多卡/分布式、streaming、DPO 和领域 benchmark 属于后续扩展。

参考：[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)、
[Chat templates](https://huggingface.co/docs/transformers/chat_templating)、
[PEFT checkpoint](https://huggingface.co/docs/peft/developer_guides/checkpoint)。

PPO/GRPO 已作为独立任务实现，不属于 Instruct SFT；见 [RL guide](rl.md)。
