# STT：语音转文字微调

[English](stt.md) | 简体中文

`stt/` 与 `tts/`、`llm/`、`diffusion/` 平级。目录以输入输出命名；ASR
（Automatic Speech Recognition，自动语音识别）用于技术说明和模型卡。
首版使用原生多语言 Whisper 全参数监督微调，从 Whisper-tiny 开始；更换并固定
Whisper-base revision 后可使用相同接口。目前仅支持单设备 fp32，不支持 LoRA、
流式识别、时间戳、说话人分离或分布式训练。

## 离线入门

[中文 Notebook](../notebooks/stt/01_finetune.zh-CN.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/stt/01_finetune.zh-CN.ipynb)

语音任务必须使用独立环境：固定 Transformers 4.57.3，不能与固定 5.18.0 的其他
任务混装。Colab 安装后若替换了已加载依赖，需要重启 runtime。

```bash
python -m venv .venv-speech
source .venv-speech/bin/activate
python -m pip install -e '.[stt,notebooks,dev]'
python -m advai_notebooks.stt validate --config configs/stt/smoke.yaml
python -m advai_notebooks.stt run --config configs/stt/smoke.yaml
```

Smoke 使用随机初始化的小型原生 Whisper 与合成 tokenizer，不下载模型权重。
八段正弦音不是对应文字的真实录音，指标与输出只证明工程链路，不代表识别质量。

## 数据契约

在忽略目录 `data/local/stt/` 保存 JSONL manifest 与音频：

```json
{"id":"clip-001","group_id":"session-001","split":"train","audio":"audio/clip-001.wav","text":"你好，欢迎使用语音识别。"}
```

每条记录必须有唯一非空 `id`、非空 `group_id`、`split`（train/validation/test）、
相对数据目录的 `audio` 和非空 `text`，三个 split 均不可为空。应按说话人、会话或
来源分组划分；group、解码后或重采样后相同音频不可跨 split。哈希检查不能识别近重复。
音频必须为 SoundFile 可解码的单声道、有限且归一化的 PCM（如 WAV/FLAC），显式重采样
至 16 kHz，不自动混合声道或截断。真实 Whisper 最长 30 秒，tiny encoder 最长 1 秒。
绝对路径、`..`、越界符号链接及超出 label token 预算的转写会被拒绝。
当前实现将音频与特征准备到内存，并不是流式数据加载器。

## 真实实验与恢复

创建 `configs/local/`，复制 `configs/stt/hf.yaml` 到该目录，修改 manifest、输出路径
与语言。模板固定 Whisper-tiny revision，默认 Chinese；英文数据改为 English。
首版每个实验必须指定一个转写语言，保证训练标签与生成前缀一致，尚未实现逐条多语言标签。
调整 token 预算时检查原生 decoder 上限。`validate` 会加载真实权重。

```bash
python -m advai_notebooks.stt validate --config configs/local/stt.yaml
python -m advai_notebooks.stt run --config configs/local/stt.yaml
python -m advai_notebooks.stt run --config configs/local/stt.yaml \
  --resume outputs/stt/whisper-tiny/latest
```

恢复路径必须是原配置 `output_dir/latest`，仅在 epoch 边界恢复，配置、数据、源码、
依赖或设备变化会被拒绝。已验证的是 CPU smoke；真实预训练模型 GPU 微调及免费 Colab
显存适配仍需在目标环境验证。Colab 长期数据和 checkpoint 应保存到 Google Drive。

## 评估与选择

报告包含 token NLL、自回归转写的 WER/CER、编辑/参考计数与逐条预测。默认 NFKC、
小写和折叠空白，保留标点。CER 去除空白；WER 按空白切词，不是中文分词器，未分词
中文优先 CER。指标使用全语料编辑次数除以参考长度，不平均单条百分比；插入可使指标大于 1。

Validation 根据配置 cer/wer/nll 的最小值选择 checkpoint，也包含 epoch-zero 基线。
`export/` 为最佳选择，`latest_export/` 为最后一轮。Test 仅报告，不参与选择。
`report.json` 包含转写内容，可能敏感，应与数据一样保密。

```bash
python -m advai_notebooks.stt evaluate --config configs/stt/smoke.yaml \
  --checkpoint outputs/stt/smoke/export --split test \
  --output outputs/stt/smoke/evaluation.json
python -m advai_notebooks.stt export --checkpoint outputs/stt/smoke/export \
  --output outputs/stt/smoke/copied-export
```

## 重载、推理与发布

```python
from advai_notebooks.stt.model import STTModel
model = STTModel.load("outputs/stt/smoke/export", "cpu")
print(model.transcribe("data/sample/stt/audio/one.wav"))
```

CLI `transcribe --checkpoint ... --input ... --output ...` 接收音频路径字符串或
`{"audio":"path.wav"}` 的 JSON 列表，输出 `{"text":"..."}` 列表。导出包含原生权重、
tokenizer、特征提取器、生成配置与项目元数据，不依赖原始数据重载。

`publish --export-dir ... --repo-id owner/model --dry-run` 预览白名单推理包，不包含
训练状态或原始录音。发布前核对模型和数据权限；项目 MIT 不覆盖第三方模型与数据许可。

## 可选 VAD 准备

[VAD](vad.zh-CN.md) 可在 STT 推理前检测并切分长录音，保留源时间偏移，遵守 Whisper
30 秒预算。检测不提供文字对齐；监督训练时不能给每个切片赋整段转录。
VAD 仍是独立安装、独立训练的任务。

参考：[Whisper 官方微调教程](https://huggingface.co/blog/fine-tune-whisper)、
[模型文档](https://huggingface.co/docs/transformers/model_doc/whisper)。
