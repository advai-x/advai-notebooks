# TTS：文字转语音微调

[English](tts.md) | 简体中文

`tts/` 与 `stt/`、`llm/`、`diffusion/` 平级。首个真实后端为
Qwen3-TTS 12Hz 0.6B Base，限定单说话人、非流式。全参数更新原生 talker 与
残差 code predictor，冻结 codec 与 speaker encoder。不支持 LoRA、多说话人、
流式训练以及 CustomVoice/VoiceDesign checkpoint。

## 离线入门与依赖

[中文 Notebook](../notebooks/tts/01_finetune.zh-CN.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/tts/01_finetune.zh-CN.ipynb)

```bash
python -m venv .venv-speech
source .venv-speech/bin/activate
python -m pip install -e '.[tts,notebooks,dev]'
python -m advai_notebooks.tts validate --config configs/tts/smoke.yaml
python -m advai_notebooks.tts run --config configs/tts/smoke.yaml
```

`qwen-tts==0.1.1` 固定 Transformers 4.57.3 / Accelerate 1.12.0，需与使用
5.18.0 / 1.15.0 的其他任务隔离；可在语音环境同时安装 STT/TTS。PyTorch 与
torchaudio 应使用匹配版本。Qwen 可能提示可选 SoX/FlashAttention 缺失；本实现
使用 SoundFile 处理本地 PCM、eager attention，不依赖这些可选后端。

`model.id: tiny` 不是 Qwen，而是预测粗量化波形帧的文本条件 GRU。正弦音不是
真实语音，只检查训练、恢复、导出和音频文件链路，其 loss 与音频不能说明语音质量。

## 单说话人数据契约

```json
{"id":"clip-001","group_id":"session-001","split":"train","speaker_id":"speaker-a","audio":"audio/clip-001.wav","text":"你好，欢迎使用语音训练教程。"}
```

全数据集只能有一个非空 `speaker_id`。每条记录需要唯一 ID、非空 group、数据相对
音频路径与非空文本，train/validation/test 均不可为空。按录音会话/来源分组，同一会话
相邻片段不能跨 split。group、解码/重采样后相同音频不能跨 split；近重复仍需人工检查。
音频应为 SoundFile 可读的有限归一化单声道 PCM，显式重采样至 24 kHz；声道、路径、
时长、text/context token 预算不满足时拒绝，不静默裁剪。

`data.reference_audio` 必须匹配训练集中的音频。固定参考的 speaker embedding
用于全部训练/验证/测试，不使用 validation/test 录音作为参考。导出保留 Base 格式与
`voice.safetensors`，推理不需要原录音或参考文本。speaker encoder 冻结，并非多说话人模型。
音频加载与 codec 编码均预先放入内存，适合有界实验，不是大规模流式训练器。

## Qwen3-TTS 训练与恢复

创建 `configs/local/`，复制 `configs/tts/qwen3.yaml`，修改 manifest、训练参考音频
与输出路径。模板固定 0.6B Base revision，最长片段 10 秒。按需调整文本/生成预算；
`max_new_tokens` 限制 codec 帧数，可能截断语音，需听测完整性后有意识地提高。

```bash
python -m advai_notebooks.tts validate --config configs/local/tts.yaml
python -m advai_notebooks.tts run --config configs/local/tts.yaml
python -m advai_notebooks.tts run --config configs/local/tts.yaml \
  --resume outputs/tts/qwen3-0.6b/latest
```

这是单设备 fp32 实现，不是上游 bf16/FlashAttention 脚本。真实模型、codec 与
optimizer 内存可能超过免费 Colab GPU，不保证适配。CPU 测试覆盖合成生命周期及小型
原生 Qwen 训练梯度，不覆盖预训练 0.6B GPU 微调、原生 codec 重建或听感质量。
Colab 使用 Drive 持久化数据/checkpoint，替换已加载依赖后重启 runtime。
恢复仅限 epoch 边界，路径必须为原 `output_dir/latest`，配置、数据、源码、依赖和设备须匹配。

## Loss 与听测

真实后端 teacher-forcing 16 个原生 codebook，显式 next-token 对齐。
Loss 为首 codebook/EOS CE + 0.3 × 残差 codebook CE，再对 utterance 等权平均；
排除 padding，保留文本投影和原生训练参数。Validation 最小 loss 选模，包括 epoch-zero，
test 仅报告。`export/` 为最佳模型，`latest_export/` 为最后一轮。

Loss 降低不代表更自然的声音。对 baseline 和 selected 使用相同未见文本听测：清晰度、
发音、说话人一致性、自然度、截断与伪影。本版提供音频生成用于听测，不提供自动 MOS
或说话人相似度评估。

```bash
python -m advai_notebooks.tts evaluate --config configs/tts/smoke.yaml \
  --checkpoint outputs/tts/smoke/export --split test \
  --output outputs/tts/smoke/evaluation.json
python -m advai_notebooks.tts export --checkpoint outputs/tts/smoke/export \
  --output outputs/tts/smoke/copied-export
```

## 重载、推理与发布

```python
from advai_notebooks.tts.model import TTSModel
model = TTSModel.load("outputs/tts/smoke/export", "cpu")
model.synthesize("你好，欢迎使用教程。", "outputs/tts/new-sample.wav")
```

输出文件必须是新路径，不覆盖旧音频。CLI `synthesize --checkpoint ... --input ...
--output ...` 接收 `{"text":"...","output_audio":"new.wav"}` 的 JSON 列表，写出
路径、采样率与时长。真实后端使用固定导出 x-vector、不使用原始参考音频上下文，greedy
生成并设置 `non_streaming_mode=True`。

导出包含原生权重、processor、generation 配置、`speech_tokenizer/`、speaker embedding
与元数据。发布白名单不包含原录音和 optimizer；`publish --export-dir ... --repo-id
owner/model --dry-run` 先检查包内容。Speaker embedding 属于衍生生物特征信息，
不因缺少原录音而匿名；训练、克隆和发布均需获得明确说话人授权，默认使用私有仓库。
项目 MIT 许可不覆盖第三方模型与数据许可。

参考：[Qwen 官方单说话人微调](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)、
[固定版本 0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc)。
