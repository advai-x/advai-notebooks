# VM — World Model

VM 在本项目明确表示 **World Model**。已实现动作条件、数值状态空间的 residual MLP：
当前 observation + action → 下一 observation，并递归预测多步未来状态。
支持从随机 tiny dynamics 训练，也支持从本项目导出的已有 VM 权重继续微调。
这是小型 dynamics 基线，**不是视频世界模型、Dreamer 或机器人规划系统**。

## 开始运行

```bash
python -m pip install -e '.[vm,notebooks,dev]'
python -m advai_notebooks.vm validate --config configs/vm/smoke.yaml
python -m advai_notebooks.vm run --config configs/vm/smoke.yaml
# 以刚才的 export 为基座，新输出目录继续微调
python -m advai_notebooks.vm run --config configs/vm/finetune.yaml
```

[Notebook](../notebooks/vm/01_finetune.ipynb) 已执行初次训练、原生权重微调与 rollout。
Tiny 合成轨迹使用 `next_state = state + action`；指标只用于工程回归，不证明世界建模能力。

## Episode 与时序对齐

JSONL 一行一个 episode：N 个 observation/timestamp，对应 N-1 个 action。
第 t 个 action 连接 observation[t] 与 observation[t+1]。

```json
{"id":"trajectory-1","group_id":"session-1","split":"train","timestamps":[0.0,0.1,0.2],"observations":[[0.0,0.0],[0.1,0.0],[0.1,0.2]],"actions":[[0.1,0.0],[0.0,0.2]]}
```

- 必须有非空 train/validation/test，ID 唯一；group 和完全重复轨迹不能跨 split。
- observation/action 维度与模型一致，所有值有限；至少两个 observation。
- timestamps 严格递增，间隔匹配 `1 / data.control_frequency_hz`；当前固定时间步。
- 状态/动作维度名称和单位、控制频率是模型契约，保存在 export；不支持任意图片或视频输入。
- horizon 窗口不能跨 episode，尾部通过 mask 排除。
- 归一化只统计 train 中每个原始 observation/action 一次，不重算重叠窗口统计。

## 模型与训练目标

将 observation/action 标准化后拼接，MLP 预测 observation delta；`predict_delta: false`
可改为直接预测标准化下一状态。训练从当前真实 observation 出发，后续递归使用自己的预测，
而不是把真实未来状态喂给模型。Loss 为有效 horizon/状态维度上的 masked MSE。
梯度累积按有效状态标量数加权。

评估报告包含原始单位的 `state_mse`（单步）与 `rollout_mse`（多步）；
条件为预测状态 + 给定真实动作序列，**不是闭环动作规划**。
默认用 validation rollout MSE 最小的 epoch 选模，含 baseline；test 不参与选模。

## 微调与断点恢复

`model.id` 可设为本项目 VM 的本地 export 或 Hub 仓库；Hub 基座必须固定 revision。
observation/action 维度、hidden_size、predict_delta 必须与基座一致。
保留基座归一化，防止改变已学习权重的物理含义；数据名称/单位/频率必须匹配。
更改训练数据或参数应创建新实验，而不是使用 `--resume`。

```bash
python -m advai_notebooks.vm run --config configs/vm/smoke.yaml \\
  --resume outputs/vm/smoke/latest
python -m advai_notebooks.vm evaluate --config configs/vm/smoke.yaml \\
  --checkpoint outputs/vm/smoke/export --output outputs/vm/test.json
```

单进程、单设备 fp32，checkpoint 保存优化器/RNG；恢复要求配置、数据、源码与相关依赖不变。

## Predict、rollout 和推理包

```python
from advai_notebooks.vm.model import VMModel
model = VMModel.load("outputs/vm/smoke/export")
next_state = model.predict([0.0, 0.0], [0.1, 0.0])
future = model.rollout([0.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
# future["observations"]: len(actions) × observation_dim
```

动作序列必须非空且不超过模型的 horizon。
CLI `rollout --checkpoint ... --input requests.json --output future.json` 输入 JSON 数组：
`[{"observation":[0.0,0.0],"actions":[[0.1,0.0],[0.0,0.1]]}]`。

`export/` 为最佳模型，`latest_export/` 为最后一轮。推理包由 safetensors 权重与包含
架构、归一化、状态/动作语义的 metadata 构成，不需要训练数据。

```bash
python -m advai_notebooks.vm export --checkpoint outputs/vm/smoke/export \\
  --output outputs/vm/copied
python -m advai_notebooks.vm publish --export-dir outputs/vm/copied \\
  --repo-id your-org/your-world-model --dry-run
```

预览会实际重新加载推理包，但不访问 Hub；不上传轨迹、优化器或日志。

## 验证边界

CPU regression 覆盖 episode 对齐、padding mask、train-only normalization、多步递归梯度、
推理重载、已有 VM 微调、严格恢复与发布预览。没有验证 CUDA/MPS、真实机器人/游戏环境、
视频/图像预测、latent stochastic dynamics、reward/done prediction 或规划能力。
真实应用需检查 Markov 状态假设、不同 horizon 的累积误差及分布外轨迹；本基线不维护循环隐状态。
