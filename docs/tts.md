# TTS: text-to-speech fine-tuning

English | [简体中文](tts.zh-CN.md)

`tts/` is a peer of `stt/`, `llm/` and `diffusion/`. The first real backend is
**Qwen3-TTS 12Hz 0.6B Base**, single speaker and non-streaming. Full fine-tuning
updates the native talker and residual code predictor. The codec and speaker
encoder are frozen. LoRA, streaming, multi-speaker training, CustomVoice checkpoints
and VoiceDesign checkpoints are not supported.

## Offline tutorial and dependencies

[Notebook](../notebooks/tts/01_finetune.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/tts/01_finetune.ipynb)

```bash
python -m venv .venv-speech
source .venv-speech/bin/activate
python -m pip install -e '.[tts,notebooks,dev]'
python -m advai_notebooks.tts validate --config configs/tts/smoke.yaml
python -m advai_notebooks.tts run --config configs/tts/smoke.yaml
```

`qwen-tts==0.1.1` requires Transformers 4.57.3 / Accelerate 1.12.0. Install speech
in a separate environment from tracks pinning 5.18.0 / 1.15.0. STT and TTS extras
can be combined in the speech environment. Use matching PyTorch/torchaudio builds
for your hardware. Qwen may print optional SoX/FlashAttention warnings; this guide
uses SoundFile for local PCM audio and eager attention, not those optional backends.

`model.id: tiny` is **not Qwen**: it is a small text-conditioned GRU predicting
coarsely quantized waveform frames. Synthetic sine waves are not spoken words.
This backend only verifies training/recovery/export/audio-file plumbing. Its
teacher-forcing loss and generated audio say nothing about speech quality.

## Single-speaker audio/text contract

```json
{"id":"clip-001","group_id":"session-001","split":"train","speaker_id":"speaker-a","audio":"audio/clip-001.wav","text":"Hello, welcome to this tutorial."}
```

Use one nonempty `speaker_id` for the entire dataset. Unique IDs, nonempty groups,
all three splits and dataset-relative paths are mandatory. Split by independent
recording session/source; adjacent segments from one session stay together.
Groups and identical decoded/resampled audio cannot cross splits. Near duplicates
still require manual checking. SoundFile-readable audio must be normalized finite
mono PCM; it is resampled to 24 kHz. Stereo, escaping paths, invalid durations and
over-budget text/context are rejected, not silently cropped.

`data.reference_audio` must match a clip **in the training split**. Its fixed speaker
embedding conditions every training/validation/test utterance. Validation/test
recordings never provide voice references. The export retains Base format plus
`voice.safetensors`, so inference needs no original reference audio or transcript.
The speaker encoder is frozen; fine-tuning does not turn this into a multi-speaker model.

The loader and codec preparation are eager in-memory operations. Start with a
bounded, consented dataset; this first guide is not a large-scale streaming trainer.

## Train Qwen3-TTS Base

Create `configs/local/` and copy `configs/tts/qwen3.yaml` there. The template pins
a 0.6B Base revision, uses 10-second maximum clips, and expects your own manifest
plus `audio/training-reference.wav`. Adapt output, text/generation limits and
duration to your dataset. Inference `max_new_tokens` caps codec frames and can cut
off an utterance; listen for completion and increase deliberately.

```bash
python -m advai_notebooks.tts validate --config configs/local/tts.yaml
python -m advai_notebooks.tts run --config configs/local/tts.yaml
python -m advai_notebooks.tts run --config configs/local/tts.yaml \
  --resume outputs/tts/qwen3-0.6b/latest
```

This is single-device fp32 training, not Qwen's upstream bf16/FlashAttention script.
It prioritizes the existing project lifecycle. Full model/codec/optimizer memory
can exceed free-Colab GPUs; no hardware fit guarantee is made. CPU tests cover
the synthetic lifecycle and small native Qwen training gradients, not pretrained
0.6B GPU fine-tuning, native codec reconstruction or perceptual quality.
Persist data/checkpoints on Drive in Colab; restart after replacing loaded dependencies.

Epoch-boundary resume restores optimizer/RNG and requires unchanged config, data,
device, source and relevant dependencies. Use the original `output_dir/latest`.

## Loss, checkpoint selection and listening

The real backend teacher-forces native 16-codebook targets with explicit next-token
alignment. The objective is first-codebook/EOS CE + 0.3 × residual-codebook CE,
averaged equally over utterances. Padding is excluded. Text projection and all
trainable talker/code-predictor parameters remain in the native export.

Validation chooses the lowest loss, including epoch zero. Test is report-only.
`export/` holds the selected model, `latest_export/` the final epoch. A lower loss
does **not** prove more natural speech. Compare baseline and selected exports on
the same unseen text list: intelligibility, pronunciation, speaker consistency,
naturalness, truncation and artifacts. This release provides waveform generation
for that listening protocol, not an automated MOS or speaker-similarity evaluator.

```bash
python -m advai_notebooks.tts evaluate --config configs/tts/smoke.yaml \
  --checkpoint outputs/tts/smoke/export --split test \
  --output outputs/tts/smoke/evaluation.json
python -m advai_notebooks.tts export --checkpoint outputs/tts/smoke/export \
  --output outputs/tts/smoke/copied-export
```

## Reload and synthesize

```python
from advai_notebooks.tts.model import TTSModel
model = TTSModel.load("outputs/tts/smoke/export", "cpu")
model.synthesize("Hello, welcome to this tutorial.", "outputs/tts/new-sample.wav")
```

Output files must be new: existing audio is never overwritten. CLI `synthesize`
accepts a JSON list of `{"text":"...","output_audio":"new.wav"}` via `--input`,
and writes audio paths/sample rates/durations via `--output`. The real backend
generates with a fixed exported x-vector, no in-context reference audio, greedy
decoding and `non_streaming_mode=True`.

Exports include native weights/processor/generation settings, native
`speech_tokenizer/` assets, speaker embedding and project metadata. Publication
is whitelisted; raw recordings and optimizer state are excluded.
`publish --export-dir ... --repo-id owner/model --dry-run` previews the package.
The exported speaker embedding is **derived biometric information**, not anonymous
because raw audio is absent. Obtain explicit speaker permission for training,
voice cloning and publication; use private repositories by default. Check model
and dataset licenses separately from the project's MIT license.

## Backend references

- [Qwen3-TTS official single-speaker fine-tuning](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [Pinned 0.6B Base model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc)
- [Qwen3-TTS source and inference API](https://github.com/QwenLM/Qwen3-TTS)
