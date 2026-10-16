# Notebook 教程

[English](README.md) | 简体中文

教程调用与 CLI 相同的实现，覆盖数据校验、训练、评估、导出/重载和推理。
英文是默认版本，以下为中文版；每本 Notebook 首格提供语言切换链接。
两版执行相同的模型流程，模型输入和必需语言前缀不随文档翻译改变。

- [Embedding](embedding/01_finetune.zh-CN.ipynb)：MNRL 与检索评估。
- [Ranker](ranker/01_finetune.zh-CN.ipynb)：BCE 与独立排序评估。
- [Decision](decision/01_finetune.zh-CN.ipynb)：原生 encoder、选项评分和校准。
- [LLM Base](llm/base/01_finetune.zh-CN.ipynb) / [Instruct](llm/instruct/01_finetune.zh-CN.ipynb)：继续预训练与 assistant SFT。
  原生 MoE 补充教程：[Base CPT](llm/base/02_moe_finetune.zh-CN.ipynb) /
  [Instruct SFT](llm/instruct/02_moe_finetune.zh-CN.ipynb)，含 expert LoRA 与路由诊断。
- [Diffusion](diffusion/01_finetune.zh-CN.ipynb)：Latent 文生图 full/LoRA。
- [STT](stt/01_finetune.zh-CN.ipynb)：原生 Whisper、WER/CER、转写与恢复。
- [VAD](vad/01_finetune.zh-CN.ipynb)：Silero decoder 微调、恢复与仅验证集校准；
  另有[离线检测与切分](vad/02_detect_and_segment.zh-CN.ipynb)教程。
- [TTS](tts/01_finetune.zh-CN.ipynb)：合成 smoke、Qwen3-TTS Base 指南、单说话人语音生成。
- [VLM](vlm/01_finetune.zh-CN.ipynb)：原生 LLaVA 图片问答、full/LoRA、processor 重载。
- [VLA](vla/01_finetune.zh-CN.ipynb)：CLIP + state 连续动作 chunk、episode mask 与物理单位。
- [VM](vm/01_finetune.zh-CN.ipynb)：数值 World Model、原生权重微调与递归 rollout。

[可选检索对照](workflows/01_retrieval_comparison.zh-CN.ipynb) 组合独立训练的 Embedding 与 Ranker。
随机 tiny 模型和 CPU 合成样例仅检查接口，不证明真实质量。
Colab 按任务安装对应 extra；长期训练需使用持久化数据/checkpoint 存储。
