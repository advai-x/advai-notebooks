# VAD：语音活动检测

[English](vad.md) | 简体中文

`vad/` 与 `stt/`、`tts/` 平级。它检测音频中“哪些时间段有人说话”，不识别文字、
不区分说话人，也不分离或降噪。真实后端适配 **Silero VAD 的 8 kHz 或 16 kHz 分支解码器**，
STFT 与编码器冻结。这是 decoder 微调，不是全参数训练或 LoRA。
第一版仅提供离线检测与切分，不实现实时端点检测、说话人分离或降噪。

## 从可运行教程开始

- [微调 Notebook](../notebooks/vad/01_finetune.zh-CN.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vad/01_finetune.zh-CN.ipynb)
- [检测、可视化与切分](../notebooks/vad/02_detect_and_segment.zh-CN.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vad/02_detect_and_segment.zh-CN.ipynb)

```bash
python -m venv .venv-vad
source .venv-vad/bin/activate
python -m pip install -e '.[vad,notebooks,dev,publish]'
python -m advai_notebooks.vad validate --config configs/vad/smoke.yaml
python -m advai_notebooks.vad run --config configs/vad/smoke.yaml
```

VAD 不依赖 Transformers 或 Qwen，可加入现有语音环境而不改变这些任务的固定版本。
使用 SoundFile/SciPy 读取和重采样，不需要 TorchAudio，也不通过 `torch.hub` 下载执行代码。

Smoke 使用随机小型特征映射与循环解码器，**不是预训练 Silero**。
八个确定性合成音频及人为区间标签仅验证工程链路；正弦音不是真人语音。
Smoke 指标与图表不代表检测质量，不要用这些样例微调预训练模型后直接部署。

## 音频与区间数据契约

在被 Git 忽略的 `data/local/vad/` 保存私有音频与 JSONL manifest：

```json
{"id":"clip-001","group_id":"session-001","split":"train","audio":"audio/clip-001.wav","speech_segments":[{"start":0.32,"end":1.84},{"start":2.10,"end":3.50}]}
```

每行必须包含唯一 `id`、非空 `group_id`、`train` / `validation` / `test` split、
相对数据集的音频路径以及 `speech_segments`。起止时间单位为秒，基于原片段零点，
须有限、排序、不重叠且不超出音频时长。`[]` 明确表示纯非语音片段。
必须标注片段内**全部语音**：未标注区域视为负例，而不是未知区域。
统一记录呼吸、笑声、歌声等是否算语音的标注政策；保留音乐、背景噪声和静音负例。

音频须为有限、归一化单声道 PCM（WAV/FLAC）。显式重采样至配置的 8 kHz / 16 kHz；
拒绝立体声而不是隐式混音。超出训练时长直接报错，不静默截断。
真实配置默认最长 30 秒；可显式提高训练上限到 300 秒。本指南在内存缓存冻结特征，
不是无限长音频或流式数据集实现。

先按原始录音、会话、说话人划分 split，再切片。同 group 和相同解码/重采样音频
不能跨 split；内容指纹无法发现近重复。每个 split 须同时有语音与非语音 frame，
但单条纯非语音录音合法。标签使用 32 ms 窗口的语音占比多数判定；
最后不完整 frame 的 padding 不参与标签或 loss 权重。准确的边界标注很重要。

## 微调真实 Silero

复制 `configs/vad/silero.yaml` 到被忽略的 `configs/local/vad.yaml`，改为自己的授权、
完整标注数据。配置固定 `silero-vad==6.2.3` 及随包 JIT 的 SHA-256，版本或权重
不符时提前失败。原生导出重载不需要寻找原安装包资产。

```bash
python -m advai_notebooks.vad validate --config configs/local/vad.yaml
python -m advai_notebooks.vad run --config configs/local/vad.yaml
python -m advai_notebooks.vad run --config configs/local/vad.yaml \
  --resume outputs/vad/silero-16k/latest
```

Resume 使用自己的 `output_dir/latest`。单设备 fp32，按有效 frame 时长加权 BCE，
支持 Adam/AdamW。循环状态每条录音重置，不跨样本；编码器/STFT 不接收梯度。
恢复限 epoch 边界，配置、数据、源码、环境或解析后的设备变化会被拒绝。
Colab 使用 Drive 持久化数据与 checkpoint。

真实 8 kHz / 16 kHz decoder 在 CPU 合成输入上验证了梯度更新与原生导出一致性，
**不代表**目标场景语音质量、GPU 微调、免费 Colab 兼容性或实时延迟已验证。
部署前须在真实 held-out 录音上评估。

## 评估与仅使用验证集的校准

验证集 BCE 选择 checkpoint，包含 epoch-zero 基线；test 仅报告。
选择后可在显式阈值网格中按**验证集区间时长 F1**校准。
退出阈值为进入阈值减去 hysteresis，最低为零；并列时选更高进入阈值。
Test 标注不参与搜索。

报告包含按 frame 时长加权的 Precision / Recall / F1、漏检率
`FN / (TP + FN)`、误报率 `FP / (FP + TN)`、BCE 与按秒计的混淆统计。
`segment_*` 指标比较实际后处理区间与人工区间，**不使用边界容差 collar**。
按全语料累计分子分母，不平均每条录音百分比。Frame 标签量化到 32 ms，
不能与区间指标混用；零分母比率记为零。

`export/` 保存选中权重及校准后的后处理参数；`latest_export/` 保存最终 epoch
及原配置阈值。`calibration.json` 记录搜索。最终报告用同一选中阈值政策对比基线与
选中权重；该政策来自选中模型的验证集，并非为基线单独调优。

```bash
python -m advai_notebooks.vad evaluate --config configs/vad/smoke.yaml \
  --checkpoint outputs/vad/smoke/export --split test \
  --output outputs/vad/smoke/evaluation.json
python -m advai_notebooks.vad export --checkpoint outputs/vad/smoke/export \
  --output outputs/vad/smoke/copied-export
```

## 离线检测与切分

```bash
python -m advai_notebooks.vad detect --checkpoint outputs/vad/smoke/export \
  --audio data/sample/vad/audio/one.wav --output outputs/vad/smoke/detection.json
python -m advai_notebooks.vad segment --checkpoint outputs/vad/smoke/export \
  --audio data/sample/vad/audio/one.wav --output-dir outputs/vad/smoke/chunks \
  --max-chunk-duration 30
```

检测返回 32 ms 概率与以秒表示的语音区间。双阈值滞回保持语音状态，合并配置范围内
的短静音，过滤短语音段，增加有界 padding，并合并接触的 padded 区间。
保存的政策使用本项目离线后处理器，不是 Silero 的 `get_speech_timestamps`。
这些参数共同影响边界质量，需要一起验证。默认检测上限 300 秒，可在保存的
`max_inference_duration` 中显式提高至最多一小时。

多文件使用 `detect-batch --checkpoint ... --input inputs.json --output ...`，
输入为音频路径或 `{"audio":"path.wav"}` 对象的 JSON 列表。
推理命令支持 `--revision` 与 `--local-files-only` 加载项目 Hub 导出。

切分只写入**全新目录**。每段保存原录音时间轴的 `source_start` / `source_end`，
超过 chunk 上限的区间继续切分，不丢样本；固定时长不保证完整句子边界。
未检测到语音时生成空 chunk 列表，不回退成整段音频。
私有录音和生成切片仍被 Git 忽略；报告含敏感时序与路径，应保持私有。

```python
from advai_notebooks.vad.model import VADModel
model = VADModel.load("outputs/vad/smoke/export", "cpu")
detection = model.detect("data/sample/vad/audio/one.wav")
print(detection["speech_segments"])
```

## 可选 STT / TTS 联动

STT 推理可采用：长音频 → VAD 切片 → 转写 → 按源时间偏移合并。
Whisper 切片不超过其 30 秒音频预算，但切分并不提供词级时间对齐。
Manifest 故意**不生成转录标签**。不得把整段转录赋给每个切片；监督训练前先重新
标注或对齐。本指南不自动合并转录文本。

TTS 数据准备可借助 VAD 检查首尾静音，但不自动删除句中停顿或呼吸，避免破坏
文字/音频对齐和韵律。STT、TTS 均不把 VAD 作为必需依赖。

## 原生导出、发布与许可

真实导出包含 `silero_vad.jit`、项目元数据及上游 MIT 声明。选中 decoder 写回原生
JIT，另一采样率分支保持不变。在测试输入上，原生 `audio_forward` 概率与项目 wrapper
一致，但直接调用原生包需另行应用元数据中的校准与后处理政策。
Tiny 导出使用 `model.safetensors`。仅加载可信 JIT/checkpoint。

`publish --export-dir ... --repo-id owner/model --dry-run` 预览白名单推理包，
不上传原录音、manifest 或 optimizer state。发布需明确执行，并检查授权及数据许可。

## 参考

- [Silero VAD 与许可](https://github.com/snakers4/silero-vad)
- [官方 decoder 微调接口](https://github.com/snakers4/silero-vad/tree/master/tuning)
- [固定安装包版本](https://pypi.org/project/silero-vad/6.2.3/)
