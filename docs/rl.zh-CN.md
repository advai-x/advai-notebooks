# LLM PPO 与 GRPO

[English](rl.md) | 简体中文

> 本页记录既有 demo 发布与验证历史，不表示已重新检查当前 Hub 的可见性、分支或标签状态。

PPO 和 GRPO 是两个独立的在线强化学习任务，分别位于 `src/advai_notebooks/llm/ppo/`
和 `src/advai_notebooks/llm/grpo/`。它们不会导入 Base、Instruct 或其他任务模块。
两者使用本项目的 PyTorch 训练循环，HF 原生 causal LM 与 PEFT 负责网络和序列化。
Instruct 仍然是 SFT；新增 RL 不改变原有训练方式。

## 运行

```bash
python -m pip install -e '.[llm-ppo,llm-grpo,dev]'
python -m advai_notebooks.llm.ppo validate --config configs/llm/ppo/smoke.yaml
python -m advai_notebooks.llm.ppo run --config configs/llm/ppo/smoke.yaml
python -m advai_notebooks.llm.grpo run --config configs/llm/grpo/smoke.yaml
```

命令别名：`advai-llm-ppo`、`advai-llm-grpo`。两个任务分别提供 `smoke.yaml`、`lora.yaml`、
`hf.yaml`，后者需要自己的数据；未进行真实领域训练或 GPU 性能验证。
`model.id` 可以指向 HF 模型或本地 full/merged 导出，包括 Instruct 的 SFT 产物。
Adapter-only 不能直接作为初始模型，请先合并导出。这是可选的文件衔接，没有模块依赖。

## 数据与奖励

每条 JSONL 样本提供 prompt 或 messages，不能同时提供；messages 可有首条 system，随后
user/assistant 交替，以 user 结束。`reference` 用于算奖励，不参与 prompt 渲染或 teacher forcing。

```json
{"id":"math-1","group_id":"problem-1","split":"train","messages":[{"role":"user","content":"compute 1 and 1"}],"reference":"2","metadata":{"domain":"math"}}
```

必须有 train / validation / test，拒绝重复 ID、跨 split 的相同问题或相同 group_id。
需要调用者以 group_id 表达改写问题、同源任务等关系。编码拒绝超过 context budget 的 prompt，
不会悄悄截断。实际预算为 prompt tokens + max(训练采样预算, 评估生成预算)。

任务各自的 `reward.py` 提供可加权的 exact match、token overlap F1、JSON validity、
目标长度接近度。Exact match 和 overlap 会 lower-case 并合并空白，至少需要非空 reference。
长度按解码文本的空白词数计算，适合 smoke 或简单规则，业务奖励应按场景替换。
配置中不存在隐式奖励模型；可以接入自定义函数，内部调用规则、判题器或训练好的 reward model：

```yaml
reward:
  exact_match_weight: 0.0
  custom_function: my_rewards:score
```

```python
# my_rewards.py，确保模块在 Python 路径中；函数需确定性、返回有限 float。
def score(row: dict, completion: str) -> float:
    return float(completion.strip() == row["reference"].strip())
```

自定义函数与启用的内置奖励相加，可以返回负数；奖励不是监督标签的交叉熵。
回调只应读取当前 row/completion，避免访问 held-out 数据；其模型、依赖和服务版本需由调用方固定。

## 算法差异

| 项目 | PPO | GRPO |
|---|---|---|
| Policy | 可配置 causal LM，full 或 LoRA | 可配置 causal LM，full 或 LoRA |
| Critic | 共享 actor decoder 表示的独立线性价值头 | 无价值头 |
| 每问题采样 | 每次 rollout 一条 completion | `group_size` 条，至少 2 条 |
| Advantage | token-level GAE；可归一化 | 同问题组内奖励减均值、除总体标准差 |
| Policy objective | token ratio clipped surrogate | token ratio clipped surrogate |
| Value objective | clipped value regression | 无 |
| Reference KL | 在每个 token 的奖励中扣除 sampled log ratio | loss 中加入 `expm1(ref_logp - logp) - (ref_logp - logp)` |
| 零方差奖励 | GAE 仍依赖 critic 与 reference | 组内 advantage 为 0；KL 仍生效 |

PPO 的 critic 与 actor 共享 backbone，价值回归也能更新 actor 的可训练参数；它不是单独复制一份
完整 value model。LoRA 模式只更新 adapter 和价值头。价值头使用动作前状态的 hidden representation。
最终任务奖励加在 completion 的最后一个有效 token；EOS 和达到生成预算都视为 episode 结束，
末端 value bootstrap 为 0。改变这个终止约定或 critic 结构，只需修改 PPO 任务。

两者使用冻结的初始 reference policy。每批重新采样并缓存旧策略 log probabilities；该批执行
`update_epochs` 次优化，随后丢弃旧 rollout。GRPO 的组内计算在 minibatch 切分前完成。
Loss 按每条 completion 的有效 token 均值，再按 completion 均值聚合，保持不同长度样本的权重一致。
`gradient_accumulation` 累积该 rollout 内的优化 minibatch，最后不足整组的窗口按实际样本数加权。

训练采样为 temperature-scaled 全词表 categorical sampling，关闭 dropout，不使用 top-k/top-p
过滤，确保采样和更新的 log probabilities 一致。`algorithm.temperature/max_new_tokens`
控制训练采样；`generation.*` 控制公开推理。评估固定 greedy，并使用 `generation.max_new_tokens`。
共享 `llm/rollout.py` 只处理采样、position IDs、log probabilities 和 EOS mask，不包含奖励、
advantage、loss、指标或选模；这些实现分别归任务所有。

原始算法来源：[PPO](https://arxiv.org/abs/1707.06347)、
[DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)。这里实现 outcome-supervised GRPO，
组内 std 使用总体标准差；没有 process reward、迭代 reference 更新或其他 GRPO 变体。

## 评估、保存与恢复

生命周期：数据/预算校验 → baseline validation/test → 在线采样与更新 → 每 epoch validation
→ 按 validation reward_mean 最大选模 → held-out test → 导出。Test 不参与参数更新或选模。
Epoch 0 也可胜出；reward 相同则选择较晚的 epoch。Tiny 随机模型和合成规则仅验证流程，
不能以 smoke 的奖励宣称推理能力提升。

```text
outputs/llm/{ppo,grpo}/smoke/
  config.json environment.json report.json
  latest.json
  checkpoints/checkpoint-N/
    policy/                  # actor 原生权重、tokenizer、metadata；PPO 附价值头
    reference/               # 冻结初始 reference，也保存其加载契约
    state.pt                 # optimizer、Python/NumPy/Torch/CUDA/MPS RNG
    resume.json              # epoch、history、选模状态、配置/数据/源码/初始权重签名
  export/                    # validation 选出的推理模型，可能是 epoch 0
  latest_export/             # 最后一轮实际训练后的模型
```

Full 导出可以用 HF `AutoModelForCausalLM.from_pretrained` 加载 actor。PPO 的价值头保存为独立
`value_head.safetensors`，生成时不使用，任务门面可完整加载。LoRA 导出为 adapter，并记录固定的
Hub base revision 或携带本地 `base_model/` 快照。原生 PEFT 加载本地 bundled base 时，先显式加载该目录。
两个任务也支持 `export --merged`，通过副本安全合并，得到独立的 HF actor。

```bash
python -m advai_notebooks.llm.ppo run --config configs/llm/ppo/smoke.yaml \
  --resume outputs/llm/ppo/smoke/latest
python -m advai_notebooks.llm.grpo evaluate --config configs/llm/grpo/smoke.yaml \
  --checkpoint outputs/llm/grpo/smoke/export --split test \
  --output outputs/llm/grpo/smoke/test.json
```

仅从本次实验 latest 完整 epoch checkpoint 恢复，不从推理 export 恢复优化器。
恢复校验配置、数据、任务源码、环境、tokenizer/template、初始模型实际参数及自定义 reward 模块源文件。
配置不一致会拒绝，包含 epochs；中断恢复保持原始 epochs 设置。Checkpoint 先写 `.incomplete`
目录再原子提交；磁盘写入失败留下的 `.incomplete` 需检查后删除，再恢复 latest。
学习率恒定，无 scheduler 状态；采样与 optimizer/RNG 状态均保存。训练器明确拒绝 distributed 环境，
勿用 torchrun 启动多副本写同一目录。

## 推理与发布

```python
from advai_notebooks.llm.ppo.model import PPOModel
from advai_notebooks.llm.grpo.model import GRPOModel

ppo = PPOModel.load("outputs/llm/ppo/smoke/export", device="cpu")
grpo = GRPOModel.load("outputs/llm/grpo/smoke/export", device="cpu")
print(ppo.chat([{"role": "user", "content": "compute 1 and 1"}]))
print(grpo.generate(["hello world"]))
```

`generate` 接受文本列表；`chat` 接受一组 messages。CLI 同样支持 generate/chat、Hub revision/cache
选项，以及 export/publish。`publish --dry-run` 离线校验英文模型卡和包。
发布排除 optimizer、reference checkpoint、原始数据、日志；PPO 附带小型价值头以支持完整门面加载。

## 首版边界

支持单设备 CPU/MPS/CUDA fp32、full/LoRA、梯度累积与裁剪、自定义终局奖励、原生导出与 epoch 恢复。
fp16/bf16、gradient checkpointing、QLoRA、多卡、异步/vLLM rollout、长上下文性能优化尚未实现，
不支持的配置明确拒绝。当前采样使用逐 token 全序列前向，侧重可读性与算法验证；大模型需评估内存与吞吐。
两任务均保留自己的 metrics、loss、reward、采样组织与训练文件，后续可分别调整。

## 已发布的 demo

两份模型从既有 tiny Instruct SFT 权重开始，分别完成 2 个 epoch 的 full 训练。
PPO 为 8 次 optimizer update；GRPO 为 32 次。选中 epoch 2 后发布到私有仓库，使用英文模型卡。

| 任务 | 模型固定 SHA | Dataset 固定 SHA |
|---|---|---|
| [PPO](https://huggingface.co/widebluesky/wbs-llm-ppo-demo) | `25beb2018bbd7a493a36ef5a5793c4ca149b2dec` | `5b621b1b02f8f097a2036c0aa0f915e4df5b8e31` |
| [GRPO](https://huggingface.co/widebluesky/wbs-llm-grpo-demo) | `fdbe74d2c76db0a7c43d432af70a5839e87442e7` | `3246400626f9c419fd1eaa4b36a1594b53089be2` |

模型与各自 Dataset 均有 `demo-v1` 标签；数据关联和原始记录指纹已验证。
Hub 下载后逐文件 SHA256、原生 Transformers/门面 logits、CLI/API generate/chat 输出及测试报告均与本地一致。
两份模型的测试 exact match 与 mean reward 均为 0.0，不代表有效的回答能力或 RL 效果提升。
模型卡提供安装和推理示例，完整记录见 [发布记录](releases/20261008-rl-demo-v1.json)。
