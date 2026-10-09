# VLM — Vision-Language Model

[English](vlm.md) | 简体中文

已实现原生 Hugging Face **LLaVA 单图问答**训练：图片 + 问题 → 文本回答。
支持 full 更新和语言侧 q_proj/v_proj LoRA；vision tower 可在 full 模式冻结或更新。
LoRA 模式仅更新语言 adapter，要求 `freeze_vision: true`。这是限定后端，不表示所有 VLM 通用支持。

## 开始运行

```bash
python -m pip install -e '.[vlm,notebooks,dev]'
python -m advai_notebooks.vlm validate --config configs/vlm/smoke.yaml
python -m advai_notebooks.vlm run --config configs/vlm/smoke.yaml
# 独立输出目录的 tiny LoRA 实验
python -m advai_notebooks.vlm run --config configs/vlm/lora.yaml
```

[Notebook](../notebooks/vlm/01_finetune.zh-CN.ipynb) 包含已执行的 CPU 示例与保存的输出。
默认 `model.id: tiny` 是随机 LLaVA（小型 Llama decoder + CLIP vision tower），
仅验证训练/导出链路，生成文字不代表可用的视觉助手。

## 数据契约

UTF-8 JSONL，每行一个单图问题/回答：

```json
{"id":"scene-1-qa","group_id":"scene-1","split":"train","image":"images/scene-1.png","prompt":"What color is the square?","answer":"Red."}
```

- 必须包含非空 train、validation、test；ID 唯一，group 不能跨 split。
- `image` 相对于 JSONL 所在目录，不允许绝对路径、`..` 或越界 symlink；必须能解码为 RGB。
- 同一解码图片不能跨 split；同一场景、近似图片、增强样本需人为设置同一 group。
- prompt/answer 非空。使用原生 processor 的 chat template 和 image token，不复用纯文本 LLM batching。
- prompt（含视觉 tokens）和 padding 的 labels 为 -100；只监督回答及 EOS。
- `max_len` 包括视觉 tokens。超长输入直接报错，不静默截断图像或回答；推理还预留 `max_new_tokens`。
- 当前仅一轮 user 图片问题 + assistant 回答；多图、视频、多轮对话不在 schema 内。

## 训练、选模和恢复

`validation.nll` 按有效回答 token 加权，越低越好；报告同时给出 perplexity。
最佳模型可能是 epoch 0 baseline，test 只做最终报告，不参与选模。
梯度累积按每组有效回答 token 加权，避免不同长度或最后小 batch 改变 loss 语义。

```bash
python -m advai_notebooks.vlm run --config configs/vlm/smoke.yaml \
  --resume outputs/vlm/smoke/latest
python -m advai_notebooks.vlm evaluate --config configs/vlm/smoke.yaml \
  --checkpoint outputs/vlm/smoke/export --split test --output outputs/vlm/test.json
```

恢复到最近完成的 epoch，保留优化器与 RNG；要求配置、数据/媒体指纹、源码和依赖版本一致。
首版单进程、单设备 fp32；不支持 QLoRA、量化基座、AMP、分布式训练。

## 推理与导出

```python
from advai_notebooks.vlm.model import VLMModel
model = VLMModel.load("outputs/vlm/smoke/export")
answer = model.chat("data/sample/vlm/images/cyan.ppm", "what color is the square ?")
```

CLI `chat --checkpoint ... --input requests.json --output responses.json` 输入为 JSON 数组：
`[{"image":"data/sample/vlm/images/cyan.ppm","prompt":"what color is the square ?"}]`。
图片路径在 CLI 推理时相对于当前工作目录。

`export/` 是 validation 选出的模型，`latest_export/` 是最后一轮模型。
Full 保存原生权重、config、processor、tokenizer、chat template；LoRA 保存 adapter 和 bundled base，
不依赖发布后仍存在的本地基座路径。Bundled base 会增加每轮 checkpoint 的存储量。

```bash
python -m advai_notebooks.vlm export --checkpoint outputs/vlm/lora/export \
  --output outputs/vlm/merged --merged
python -m advai_notebooks.vlm publish --export-dir outputs/vlm/merged \
  --repo-id your-org/your-vlm --dry-run
```

发布只选择推理文件并实际重载验证；不会上传原始图片、数据、优化器或日志。
只有显式移除 `--dry-run` 才会访问 Hub。实际发布前确认模型和数据许可证。

## 真实基座与验证边界

[固定 revision 的 LLaVA 配置模板](../configs/vlm/hf.yaml) 使用 `llava-hf/llava-1.5-7b-hf`。
其 processor、视觉 token 数和回答边界已单独检查；**没有执行 7B GPU 微调**。
7B fp32 权重本身约 28 GB，还需视觉模型、梯度/优化器与激活内存，不能将它视为免费 Colab 配置。
需替换自己的 JSONL 和输出路径，并按硬件预算选择兼容的原生 `model_type=llava` 基座。
本地原生 LLaVA checkpoint 也可作为 `model.id`，Hub 基座必须指定 revision。
`validate` 会加载完整基座和 processor 来检查真实输入预算，因此真实配置的校验同样需要足够内存；
在还没有目标硬件时，先用 smoke 配置检查数据契约。

离线测试覆盖 full/LoRA 训练、答案掩码、merged、processor 保存/重载、严格恢复与发布预览。
NLL 不是问答正确率；真实实验还需答案准确性、grounding 和人工审核。
不支持 Qwen-VL、LLaVA-OneVision、视频后端；CUDA/MPS 和真实业务数据尚未验证。

官方参考：[LLaVA](https://huggingface.co/docs/transformers/model_doc/llava)、
[多模态 chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)。
