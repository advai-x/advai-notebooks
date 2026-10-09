# Diffusion 图像模型

`diffusion/` 与 `embedding/`、`ranker/`、`decision/`、`llm/` 平级，独立拥有训练、指标、推理和发布。
首版支持 Stable Diffusion v1/v2 风格的 latent 文生图：单个 CLIP 文本编码器、VAE 和 UNet。
网络组件和原生保存/加载复用 Diffusers 0.41.0；训练循环、数据契约、评估与选模由本项目实现。
不导入其他任务。文本条件编码器属于 Diffusion 自己的组件，不依赖项目 Embedding 模型。

## 安装与离线示例

```bash
python -m pip install -e '.[diffusion,notebooks,dev,publish]'
python -m advai_notebooks.diffusion validate --config configs/diffusion/smoke.yaml
python -m advai_notebooks.diffusion run --config configs/diffusion/smoke.yaml
python -m advai_notebooks.diffusion run --config configs/diffusion/lora.yaml
```

两份配置分别执行 UNet full 和 LoRA。tiny 包含随机初始化的微型 UNet、CLIP 文本编码器、VAE、
本地 WordLevel tokenizer 和噪声 scheduler，不下载基座。8 张 16×16 合成图片仅用于工程验证；
VAE 和文本编码器也未预训练，因此生成结果不代表图像质量或文本对齐能力。
`run` 拒绝非空输出目录；重复运行请修改 `output_dir`，中断恢复使用下面的 `--resume`。

## 职责与输入

| 职责 | 入口 |
|---|---|
| 配置 | `config.py` |
| 网络组件 | `backbones.py`：原生 UNet/VAE/文本编码器；`heads.py`：UNet 输出层引用 |
| 加载、保存、组合与推理 | `model.py`：`DiffusionModel.load/save/generate` |
| 图片与 caption | `data.py` |
| 批次、噪声、时间步 | `sampling.py` |
| 加噪与推理调度 | `schedulers.py`，区别于学习率 scheduler |
| 目标与优化器 | `losses.py`、`optimizer.py` |
| 训练、指标与评估 | `train.py`、`metrics.py`、`evaluate.py` |
| 生命周期、恢复、CLI 与发布 | `pipeline.py`、`runtime.py`、`cli.py`、`publish.py` |

每行 JSONL：

```json
{"id":"image-1","group_id":"asset-1","split":"train","image":"images/one.png","caption":"a red square"}
```

`image` 相对 JSONL 所在目录，不允许绝对路径或越界路径。字段严格校验，三个 split 必须非空；
同一 group、相同解码 RGB 图片不能跨 split。指纹包含原始图片文件 SHA256、解码图片及记录内容。
图片进行 EXIF 方向修正、RGB 转换、居中裁剪/缩放，归一化至 [-1,1]。首版无随机图像增强。
Caption 和推理 prompt 超出文本编码器预算会拒绝，避免静默截断。Caption 相同不等于图片重复。

真实数据使用 `configs/diffusion/hf.yaml`，调整基座 ID/revision、图片分辨率和 JSONL 路径。
远端初始基座先解析成不可变 commit SHA；本地基座支持原生完整 pipeline 或本项目 full/merged 导出。
SDXL、多文本编码器、DiT/Flux、视频、img2img 和 inpainting 不在首版支持范围，不能只更换 ID 使用。

## 训练与评估

Full 更新完整 UNet；LoRA 只更新 UNet attention 的 `to_q/to_k/to_v/to_out.0` 适配层，
rank、alpha 可配置。两种方式均冻结 VAE、文本编码器；不把 full 解释为整个 pipeline 全量训练。
训练从 VAE posterior 采样 latent、采样高斯噪声和均匀时间步、加噪，使用每张图的平均 MSE。
目标由基座 scheduler 的 `prediction_type` 决定，也可显式配置 `epsilon` 或 `v_prediction`。
学习率恒定，支持 AdamW/Adam、梯度累积和裁剪、UNet gradient checkpointing。
最后不足一个累积窗口的批次按实际样本数归一化。

评估使用 VAE posterior mode；噪声和时间步由 evaluation seed、样本 ID、draw 编号共同固定。
评估逐图执行，改变训练 RNG 或 batch size 不改变评估结果。`noise_draws` 指每张图重复评估的次数。
`metrics.py` 独立实现 `denoising_mse`，提供 task/schema/name/value/n/group/direction 指标记录；
`n` 是图片数，而不是噪声抽样次数，选模方向是 minimize。

生命周期为：数据校验 → baseline validation/test → 训练 → validation 选模 → test → 原生导出。
选模包含 epoch 0；严格下降才更新 best。test 不参与优化或选模。
固定 generation seed 的 test caption 样例保存在 `samples/selected-test/`。
这些样例只作为定性检查；去噪损失下降不能证明生成质量改善，当前未实现 FID/CLIP score。
没有可选拟合后处理或校准阶段。

## 产物与中断恢复

```text
output_dir/
  baseline/                           # 固定初始模型
  checkpoints/checkpoint-N/
    model/                            # 对应 epoch 的推理资产
    state.pt                          # optimizer + Python/NumPy/Torch/device RNG
    resume.json                       # 签名、epoch、选模历史
  latest.json                         # 最近完整 checkpoint 的原子指针
  export/                             # validation 选中模型，可能是 epoch 0
  latest_export/                      # 最后完成训练的模型
  samples/selected-test/               # 固定 seed 的图像与 images.json
  report.json config.json data_summary.json environment.json
```

```bash
python -m advai_notebooks.diffusion run \
  --config configs/diffusion/smoke.yaml --resume outputs/diffusion/smoke/latest
```

只从同一实验的完整 checkpoint 或 `latest` 别名恢复，推理 export 不可恢复 optimizer。
恢复校验配置（包括计划 epoch 数）、图片内容、基座实际参数、组件配置、tokenizer、相关源码与依赖版本。
完整 epoch 先写 `.incomplete` 再原子提交，指针最后更新。未完成 epoch 从上一边界重放。
若 checkpoint 留下 `.incomplete`，先检查并清理该目录，再重放该 epoch。
完成后的恢复可重新生成 export；不覆盖已有 checkpoint。支持单进程，distributed launch 明确拒绝。

## 加载、生成和导出

```python
from advai_notebooks.diffusion.model import DiffusionModel

model = DiffusionModel.load("outputs/diffusion/smoke/export", device="cpu")
images = model.generate(["a red square pattern"], seed=123, num_inference_steps=4)
images[0].save("sample.png")
```

`generate` 返回同序 PIL 图片列表，支持 prompt 字符串或列表。seed、推理步数、guidance scale 可覆盖。
每张图使用独立 CPU generator，推理不消费训练的全局 RNG；同环境与批次顺序下可复现。
跨硬件不保证逐像素相同。DDIM/DDPM 推理 scheduler 可配置，训练始终使用保存的 DDPM 噪声过程。

```bash
python -m advai_notebooks.diffusion generate \
  --checkpoint outputs/diffusion/smoke/export \
  --input configs/diffusion/prompts.json --output outputs/diffusion/generated --seed 123
python -m advai_notebooks.diffusion evaluate \
  --config configs/diffusion/smoke.yaml --checkpoint outputs/diffusion/smoke/export \
  --split test --output outputs/diffusion/reloaded-test.json
python -m advai_notebooks.diffusion export \
  --checkpoint outputs/diffusion/lora/export --merged --output outputs/diffusion/merged
```

`generate --output` 是目录，包含 PNG 和 `images.json`。`evaluate --output` 是 JSON 文件。
Full 保存 Diffusers 原生 `model_index.json` 与各组件的 safetensors/config/tokenizer，可原生重载。
LoRA 保存原生 `pytorch_lora_weights.safetensors`（含 rank/alpha 元数据），以及冻结的 `base_model/`。
为保证离线重载，首版所有 LoRA 包均附带完整基座，体积大于仅 adapter 的包。
合并导出在加载副本上 fuse/unload LoRA，不改变训练模型；原生推理支持：

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("outputs/diffusion/smoke/export")
# Native LoRA: load export/base_model, then pipe.load_lora_weights(export).
```

LoRA 原生推理还需恢复训练后的调度配置：按照 `advai_metadata.json` 中的
`scheduler.inference`，使用 `DDIMScheduler` 或 `DDPMScheduler` 从导出的
`noise_scheduler/` 加载并赋给 `pipe.scheduler`。附带基座保留原始配置；显式改变
`prediction_type` 时，直接沿用基座会使用错误的预测目标。项目 `load/generate` 会自动恢复。

## 显式发布

```bash
python -m advai_notebooks.diffusion publish \
  --export-dir outputs/diffusion/smoke/export \
  --repo-id your-org/your-diffusion --dry-run
```

移除 `--dry-run` 才上传，默认私有；支持 `--model-card` 自定义英文模型卡。
发布前从筛选后的包实际加载。只发布原生推理组件、LoRA/基座、噪声配置和哈希 manifest，
排除图片数据、生成样例、训练状态与日志。未来 demo 命名可用 `widebluesky/wbs-diffusion-demo`，
本次只实现入口和离线验证，没有创建该 Hub 模型或 Dataset。
训练用的 Dataset 应独立发布并验证后再在模型卡关联。

## 验证边界

已有 CPU fp32 检查：full/LoRA 完整训练、参数更新和冻结、原生重载逐像素一致、alpha 保留、
合并导出、确定性评估、梯度累积、gradient checkpointing、`v_prediction` 与 DDPM 调度重载、
精确 epoch 恢复、CLI、离线发布包以及任务导入独立性。
CUDA/MPS 设备可配置，但未在目标硬件验证；首版不支持 AMP、多卡、EMA、量化与其他模型族。
真实预训练大模型和真实图片数据的质量/资源消耗需要相应硬件上的独立验证。

参考：[Diffusers 文生图训练](https://huggingface.co/docs/diffusers/training/text2image)、
[LoRA 训练](https://huggingface.co/docs/diffusers/training/lora)、
[原生 pipeline 加载](https://huggingface.co/docs/diffusers/using-diffusers/loading)。
