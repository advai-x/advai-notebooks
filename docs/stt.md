# STT: speech-to-text fine-tuning

English | [简体中文](stt.zh-CN.md)

`stt/` is an independent training track alongside `tts/`, `llm/` and `diffusion/`.
STT describes the input/output task; ASR (automatic speech recognition) is the
standard technical term used in model cards and research.

The first backend is native multilingual Whisper supervised fine-tuning. Start
with `openai/whisper-tiny`; `openai/whisper-base` can use the same interface after
pinning its revision. This implementation updates all Whisper parameters on one
device in fp32. LoRA, timestamps, diarization, streaming and distributed training
are not implemented.

## Start with the offline tutorial

[Notebook](../notebooks/stt/01_finetune.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/stt/01_finetune.ipynb)

Use a **separate environment** for speech. These extras pin Transformers 4.57.3;
several other tracks pin 5.18.0 and cannot be installed together in that environment.

```bash
python -m venv .venv-speech
source .venv-speech/bin/activate
python -m pip install -e '.[stt,notebooks,dev]'
python -m advai_notebooks.stt validate --config configs/stt/smoke.yaml
python -m advai_notebooks.stt run --config configs/stt/smoke.yaml
```

The smoke model is a small randomly initialized native Whisper with a synthetic
tokenizer. Its eight sine-wave clips are **not recordings of their transcripts**.
They test mechanics only: metrics and predictions do not establish recognition
quality. No model weights are downloaded for `model.id: tiny`.

## Audio/transcript data contract

Store a JSONL manifest and audio under an ignored directory such as `data/local/stt/`:

```json
{"id":"clip-001","group_id":"session-001","split":"train","audio":"audio/clip-001.wav","text":"你好，欢迎使用语音识别。"}
```

Every row requires a unique nonempty `id`, nonempty `group_id`, `split` (`train`,
`validation`, `test`), dataset-relative `audio` and nonempty `text`. All three
splits must be present. Split by speaker/session/source as appropriate before
writing the manifest, not by individual neighboring clips. A group and duplicate
decoded/resampled audio cannot cross splits. Hash checks do not detect near duplicates.

Audio must be finite, normalized mono PCM readable by SoundFile (e.g. WAV/FLAC).
The loader resamples explicitly to 16 kHz; it does not downmix stereo or silently
truncate long clips. Configure duration bounds up to 30 seconds; the tiny smoke
encoder supports at most 1 second. Absolute paths, `..` and escaping symlinks are
rejected. Labels exceeding `max_label_tokens` are rejected rather than truncated.
The current loader eagerly prepares clips/features in memory; this is a bounded
training guide, not a streaming dataset implementation.

## Run a real experiment

Copy `configs/stt/hf.yaml` into `configs/local/stt.yaml` (create that ignored
directory first). It pins a specific Whisper-tiny revision and Chinese transcription.
Replace the manifest, output path and language; use English for English recordings.
This first training guide requires one explicit language per experiment so training
labels and generation prefixes agree; mixed-language per-row tags are not implemented.
Check native decoder limits when changing
label/generation budgets. `validate` loads the model and may require GPU memory.

```bash
python -m advai_notebooks.stt validate --config configs/local/stt.yaml
python -m advai_notebooks.stt run --config configs/local/stt.yaml
python -m advai_notebooks.stt run --config configs/local/stt.yaml \
  --resume outputs/stt/whisper-tiny/latest
```

Use the actual `output_dir/latest` from your configuration. Resume occurs at epoch
boundaries and rejects changed config, data, source, package versions or device.
CPU smoke execution is tested; real pretrained GPU fine-tuning and free-Colab
memory fit are not established by that test. Keep data/checkpoints on Google Drive
in Colab. Restart the runtime if installing speech replaces loaded dependencies.

## Evaluation and selection

Reports include teacher-forced token NLL, autoregressively decoded WER/CER,
edit/reference counts and per-clip reference/prediction pairs. The default
normalization is NFKC, lowercase and collapsed whitespace; punctuation is preserved.
CER removes whitespace; WER splits on whitespace and is not a Chinese word segmenter.
Prefer CER for unsegmented Chinese. Rates sum edit counts over the entire corpus,
not per-utterance percentages, and can exceed 1 due to insertions.

Validation selects the lowest configured `cer`, `wer` or `nll`, including the
epoch-zero baseline. `export/` is selected; `latest_export/` is the final epoch.
Test data is report-only, never used for checkpoint selection. `report.json` may
contain sensitive transcripts; keep it private with the dataset.

```bash
python -m advai_notebooks.stt evaluate --config configs/stt/smoke.yaml \
  --checkpoint outputs/stt/smoke/export --split test \
  --output outputs/stt/smoke/evaluation.json
python -m advai_notebooks.stt export --checkpoint outputs/stt/smoke/export \
  --output outputs/stt/smoke/copied-export
```

## Reload and transcribe

```python
from advai_notebooks.stt.model import STTModel
model = STTModel.load("outputs/stt/smoke/export", "cpu")
print(model.transcribe("data/sample/stt/audio/one.wav"))
```

CLI `transcribe` accepts a JSON list of audio path strings or `{"audio":"path.wav"}`
objects via `--input`, and writes a list of `{"text":"..."}` objects via `--output`.
Local exports include native weights, tokenizer, feature extractor, generation
settings and project metadata, and reload without the original dataset.

`publish --export-dir ... --repo-id owner/model --dry-run` previews a whitelisted
inference package. Training state and raw recordings are not included. Only publish
after checking dataset/model permissions; the project's MIT license does not
relicense pretrained weights or datasets.

## Backend references

- [Hugging Face Whisper fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper)
- [Whisper model documentation](https://huggingface.co/docs/transformers/model_doc/whisper)
- [Pinned Whisper-tiny model](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af)
