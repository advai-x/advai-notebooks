# 原生 MoE 训练与微调

[English](llm-moe.md) | 简体中文

Mixture of Experts（MoE）是 Decoder **架构**，不是新的训练目标。沿用 Base 的继续预训练
（CPT）与 Instruct 的 assistant-only SFT，各任务继续拥有数据、损失、指标和选模逻辑。
首版原生后端为 `olmoe`、`qwen3_moe`；其他稀疏架构明确报错，不静默忽略路由目标。

入口：[Base Notebook](../notebooks/llm/base/02_moe_finetune.zh-CN.ipynb) ·
[Instruct Notebook](../notebooks/llm/instruct/02_moe_finetune.zh-CN.ipynb) ·
[Base Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/base/02_moe_finetune.zh-CN.ipynb) ·
[Instruct Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/instruct/02_moe_finetune.zh-CN.ipynb)。
这些是架构补充教程，不新增任务 track。

## 安装与配置选择

```bash
python -m pip install -e '.[llm-base,llm-instruct,notebooks,dev,publish]'
python -m advai_notebooks.llm.base validate --config configs/llm/base/moe/smoke.yaml
python -m advai_notebooks.llm.base run --config configs/llm/base/moe/smoke.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/moe/smoke.yaml
```

已验证 SDK：Transformers 5.18.0、PEFT 0.18.1、Accelerate 1.15.0。
`configs/llm/{base,instruct}/moe/` 均包含：

| 配置 | 模型与更新策略 | 用途 |
| --- | --- | --- |
| `smoke.yaml` | 随机原生 tiny OLMoE；full，训练 router | 离线 CPU 生命周期 |
| `smoke_qwen3.yaml` | 随机原生 tiny Qwen3-MoE；full | 第二种原生后端 |
| `smoke_lora.yaml` | Tiny OLMoE attention q/v LoRA；冻结 router | 小型 adapter 流程 |
| `smoke_experts_lora.yaml` | Attention 与原生三维 expert 参数 LoRA；冻结 router | Expert 梯度与合并重载 |
| `olmoe_lora.yaml` | 固定版本 OLMoE Base / Instruct；attention LoRA | 真实模型 GPU 模板 |
| `qwen3_lora.yaml` | 固定版本 Qwen3-30B-A3B Base / 后训练模型；attention LoRA | 真实模型 GPU 模板 |

Tiny 每层四个 expert、top-2 路由，使用固定合成 tokenizer。它们是真正的稀疏模型，不是 dense
替代品，但**不是预训练模型**。合成 smoke 成功运行不证明领域质量。

## 数据与架构契约

沿用 [Base/Instruct 数据契约](llm.zh-CN.md#数据契约)：Base 是分组文档，Instruct 是纯文本对话。
先按原文档/对话分组划分，再编码；train、validation、test 都不能为空，test 不参与选模。
Base 保留文档 chunk 边界目标；Instruct 屏蔽 system/user/角色前缀，只监督助手回复和结束标记。

`model.architecture` 为 `auto`（自动检测）、`dense` 或 `moe`；显式不匹配会报错。
`tiny-olmoe`、`tiny-qwen3-moe` 选择离线后端。Hub MoE 初始化要求固定 40 位 revision SHA，
禁用远程代码；本地 full/merged 导出也可初始化训练，adapter-only 必须先合并。

真实 OLMoE Instruct 使用其固定版本原生模板。Qwen3 Instruct 有意采用显式**纯文本 non-thinking**
Jinja 模板：保留 Qwen 角色/结束 token，将空 `<think>` prefill 放在监督区外，用 generation
block 标记助手正文与结束 token，使多轮标签和推理前缀稳定。这不是 thinking/reasoning SFT，
也不支持上游动态模板的所有特性。修改模板时必须检查 token IDs、解码后的 labels 和推理 prompt。
已用真实固定版本 tokenizer 检查多轮样例，未运行真实权重训练。

## 更新策略与 LoRA 目标

Full 默认训练 router；设置 `training.moe_train_router: false` 可显式冻结 router，其他 full
参数仍可训练。首版 LoRA 默认冻结 router，拒绝 `moe_train_router: true`；不宣称支持 router LoRA。

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

固定 SDK 的 expert 权重是三维 `nn.Parameter`，不是普通 Linear module。必须使用 PEFT
`target_parameters`；只有 attention 的 `target_modules` 不会训练 expert。精确/后缀目标
必须命中原生三维专家权重，否则报错。仅 expert LoRA 可设置 `lora_target_modules: []`。
禁止将 router 列为 module 目标；通过 `validate` 和 `training.moe_gradients` 检查真正训练的参数。

PEFT 0.18.1 对这些 expert 参数 wrapper 会发出通用 unsupported-layer 警告。CPU 测试覆盖两种
tiny 后端的非零 expert-adapter 梯度、冻结 base 权重不变、合并与重载 logits 一致；这不证明
所有真实 checkpoint、SDK、CUDA kernel 都兼容。真实模板默认 attention-only LoRA；启用 expert
目标前先在目标硬件做小规模梯度/合并检查。

## 训练目标、评估与诊断

启用原生 router logits，将其 load-balancing 辅助损失纳入训练。`moe_aux_loss_coef` 可覆盖
原生系数，null 保留原生值；零系数关闭其贡献，但诊断仍要求有限原生输出。不额外引入 router z-loss。

同一梯度累积窗口中 microbatch `i`：

```text
loss_i = shifted_CE_sum_i / N_window
       + coefficient * native_aux_i * N_targets_i / N_window
N_window = sum_i N_targets_i
```

这是按监督 token 加权的 **microbatch 局部**辅助损失均值，不再乘除累积次数。
Load balancing 是非线性的，改变 microbatch 划分可能改变目标，不等同于拼成一个大 batch 后计算。

仍以 validation **NLL** 选模，包含 epoch zero。Test NLL/PPL 排除辅助损失；PPL = exp(NLL)。
只在 tokenizer、语料与监督区域兼容时比较困惑度。

报告新增独立 `moe` 字段：按监督 token 加权的原生辅助损失、各 MoE 层分配次数/比例、
最大/平均负载比、平均 router entropy（nats）。排除 padding，但统计**所有非 padding 输入 token**，
包括 SFT prompt，而非仅助手监督区。次数之和 = 输入 token × top-k；比例以分配次数为分母，
合计为一，不是独立 token 占比。层号按原生 MoE 层顺序。均匀路由不证明专家学到了有用专长或质量提升。

`training.moe_model` 记录真实总/可训练参数数、dtype 与参数字节数；`training.moe_gradients`
按 router/experts/other 记录 optimizer step、可训练张量、非零梯度 step 与最近一次裁剪后范数。
诊断不保留梯度张量，不修改 optimizer/RNG 状态。

## 恢复、导出与推理

```bash
python -m advai_notebooks.llm.base run --config configs/llm/base/moe/smoke.yaml \
  --resume outputs/llm/base/moe-smoke/latest
python -m advai_notebooks.llm.instruct export \
  --checkpoint outputs/llm/instruct/moe-smoke-experts-lora/export \
  --output outputs/llm/instruct/moe-merged --merged
```

使用实际配置的 `output_dir`。完整 epoch checkpoint 包含权重、optimizer、scheduler、RNG
与实验签名；推理导出不包含续训状态。固定 SDK 将 full MoE expert 保存为兼容的逐专家布局，
续训先用原生加载器转换，再由 Trainer 恢复其余状态，避免 raw load 遗漏专家权重。
CPU/fp32 测试对两种 tiny 后端的 full/expert LoRA 验证中断与不中断的历史、梯度诊断和最终权重一致。

Full/merged 使用 `AutoModelForCausalLM.from_pretrained`；adapter 使用 PEFT 与固定 Hub base，
或打包本地/tiny base。元数据保留 dtype、架构、辅助损失系数、router 策略。合并重载副本，
不修改活跃 adapter；生成时关闭不必要的 router 输出，保留原生生成配置和训练/推理模板。
沿用 [LLM 指南](llm.zh-CN.md) 的 generate/chat、evaluate、export、publish --dry-run。
发布须显式执行，训练不自动上传。

## 真实模型内存与范围

将真实模板复制到被忽略的 `configs/local/`，用 `--root .` 指定仓库根目录，替换私有数据与输出路径，
核查模型/数据许可证。不要用合成 fixture 微调真实模型；变更固定模型/tokenizer revision 属于新实验。

`model.dtype` 控制权重加载精度，`runtime.precision` 控制 AMP；GPU 模板显式设为 bfloat16 / bf16。
激活参数少不代表只常驻这些权重。bf16 仅权重约需 OLMoE 14 GB、Qwen3 61 GB（十进制），
另需 adapter、梯度、optimizer、logits、activation、临时 buffer；合并/导出还需要足够主存。

`max_weight_memory_gb` 在真实权重加载前用原生 meta 模型参数数做下限检查，CUDA 还检查当前可用内存。
这不是训练峰值预测或准入保证；真实 `validate` 也会加载权重，并非便宜的元数据查询。
预算报错时不要盲目提高上限；免费 Colab 不保证适配，Qwen3 激活参数少也不代表适合小型单卡。

本地已验证：原生 tiny CPT/SFT、full/attention/expert LoRA、路由辅助目标和诊断、CPU epoch
精确恢复、原生导出/重载/合并、推理与离线发布。真实 tokenizer 格式检查独立于训练测试。
真实预训练模型 GPU 训练、CUDA/AMP 恢复和领域 benchmark **尚未验证**。首版不含
distributed/expert parallel、量化/QLoRA、router LoRA、大规模从零预训练或 MoE PPO/GRPO。
仓库 MIT 许可证不替代上游模型与数据许可证。

官方参考：[OLMoE](https://huggingface.co/docs/transformers/model_doc/olmoe)、
[Qwen3-MoE](https://huggingface.co/docs/transformers/model_doc/qwen3_moe)、
[PEFT 参数目标](https://huggingface.co/docs/peft/package_reference/lora#targeting-nnparameter-directly)、
[OLMoE 模型卡](https://huggingface.co/allenai/OLMoE-1B-7B-0924)、
[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B)。
