# VLA — Vision-Language-Action Model

已实现可训练的 **CLIP + state + 连续动作 chunk head 行为克隆基线**。
视觉观测和语言指令经各自 CLIP encoder 处理，与标准化状态拼接，再预测未来 H 步动作。
可以冻结预训练编码器只训练 head，或完整更新 encoder + head。
**不是 OpenVLA / SmolVLA checkpoint 适配，也不是已验证的机器人控制器。**

## 开始运行

```bash
python -m pip install -e '.[vla,notebooks,dev]'
python -m advai_notebooks.vla validate --config configs/vla/smoke.yaml
python -m advai_notebooks.vla run --config configs/vla/smoke.yaml
# tiny encoder 一起更新
python -m advai_notebooks.vla run --config configs/vla/full.yaml
```

[Notebook](../notebooks/vla/01_finetune.ipynb) 有保存的 CPU 输出。
Tiny CLIP 和动作 head 均随机初始化；真实编码器模板在 [hf.yaml](../configs/vla/hf.yaml)，
固定到 `openai/clip-vit-base-patch32` revision，动作 head 仍从随机参数开始。

## Episode 数据契约

JSONL 一行一个 episode，episode/group 不能跨 train、validation、test：

```json
{"id":"episode-1","group_id":"session-1","split":"train","instruction":"move right","steps":[{"timestamp":0.0,"image":"images/frame-0.png","state":[0.0,0.0],"action":[0.1,0.0]},{"timestamp":0.1,"image":"images/frame-1.png","state":[0.1,0.0],"action":[0.1,0.1]}]}
```

- 三个 split 均非空；ID 唯一。相同会话/机器人轨迹的 episode 需人为设成同一 group。
- instruction 非空，所有 step 有有限 timestamp、state、action 和可解码的本地图片。
- 图片相对 JSONL，不能越界；重复解码图片不能跨 split。
- 时间戳严格递增，间隔需匹配 `1 / data.control_frequency_hz`，当前不支持不规则采样。
- `state_dim` / `action_dim`、维度名称和单位、动作坐标系、控制频率必须显式一致。
- 每个 step 的 action 是在该图像/状态之后要执行的动作；第 t 个样本预测 t 到 t+H-1。
- action chunk 不跨 episode；尾部 padding 通过 mask 排除，不进入训练/评估。
- 指令超出 `max_len` 直接报错，不静默截断；CLIP context 上限取自原生模型。

仅从 train 的每个原始 step 拟合 state/action 均值和标准差，重叠 chunk 不重复计入统计。
统计量与物理语义保存在 export 中；评估与推理使用同一统计量，不能重新拟合。

## Loss、指标和选模

训练：标准化动作空间 masked MSE。评估：反标准化后的 action MSE / MAE，
对所有有效 horizon/动作维度加权；`validation.action_mse` 越低越好，含 epoch 0 baseline。
梯度累积按有效动作标量数加权；test 不参与选模。

```bash
python -m advai_notebooks.vla run --config configs/vla/smoke.yaml \\
  --resume outputs/vla/smoke/latest
python -m advai_notebooks.vla evaluate --config configs/vla/smoke.yaml \\
  --checkpoint outputs/vla/smoke/export --output outputs/vla/test.json
```

单进程、单设备 fp32；epoch checkpoint 保留优化器/RNG，并严格检查配置、数据/媒体、源码和依赖版本。

## 动作推理、导出和发布

```python
from advai_notebooks.vla.model import VLAModel
model = VLAModel.load("outputs/vla/smoke/export")
prediction = model.predict("data/sample/vla/images/cyan.ppm", "move up", [0.0, 0.1])
# prediction["actions"]: H × action_dim，原始物理单位
# prediction["action_spec"]: 名称、单位、坐标系、控制频率、horizon
```

CLI `predict --checkpoint ... --input requests.json --output actions.json` 输入 JSON 数组：
`[{"image":"data/sample/vla/images/cyan.ppm","instruction":"move up","state":[0.0,0.1]}]`。
推理图片路径相对当前工作目录。输出不自动裁剪、不自动执行机器人动作。

`export/` 保存最佳 encoder、processor/tokenizer、动作 head safetensors、训练归一化和动作语义；
`latest_export/` 为最后一轮。重新加载不会依赖原始 CLIP 下载位置。

```bash
python -m advai_notebooks.vla export --checkpoint outputs/vla/smoke/export \\
  --output outputs/vla/copied
python -m advai_notebooks.vla publish --export-dir outputs/vla/copied \\
  --repo-id your-org/your-vla --dry-run
```

发布仅筛选推理文件并重载验证，不上传 episode/图像/训练状态；默认预览不访问 Hub。

## 验证范围与限制

CPU tests 覆盖冻结/full encoder、原生 CLIP 本地基座初始化、尾部 mask、train-only normalization、
时间/媒体/group 契约、动作推理重载、恢复与发布预览。
实际 CLIP Hub 权重的机器人微调、CUDA/MPS 和闭环机器人成功率没有执行验证。
不含离散 action tokenizer、diffusion action head、LoRA、LeRobot/RLDS adapter 或机器人连接。

离线 MSE 不能证明控制安全或任务成功率；真实部署必须增加动作边界、硬件安全控制、
坐标变换与闭环评估。机器人数据授权与基座许可证也需单独检查。

官方编码器参考：[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)。
