# VAD: voice-activity detection

English | [简体中文](vad.zh-CN.md)

`vad/` is an independent speech track alongside `stt/` and `tts/`. It identifies
speech intervals in audio, not words, speaker identities or clean speech signals.
The first real backend adapts **Silero VAD's selected 8 kHz or 16 kHz decoder**;
its STFT and encoder stay frozen. This is decoder fine-tuning, not full-model
training or LoRA. Detection and segmentation are offline; real-time endpointing,
diarization and denoising are outside this guide.

## Start with the executable tutorials

- [Fine-tune](../notebooks/vad/01_finetune.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vad/01_finetune.ipynb)
- [Detect, visualize and segment](../notebooks/vad/02_detect_and_segment.ipynb) · [Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vad/02_detect_and_segment.ipynb)

```bash
python -m venv .venv-vad
source .venv-vad/bin/activate
python -m pip install -e '.[vad,notebooks,dev,publish]'
python -m advai_notebooks.vad validate --config configs/vad/smoke.yaml
python -m advai_notebooks.vad run --config configs/vad/smoke.yaml
```

VAD does not depend on Transformers or Qwen. Its extra can be added to the existing
speech environment without changing those task pins. SoundFile/SciPy handle audio
I/O; the guide does not need TorchAudio or download code through `torch.hub`.

The smoke backend is a small random feature projection and recurrent decoder,
**not pretrained Silero**. Eight deterministic tone/noise clips and their artificial
interval labels exercise the lifecycle only. Tones are not human speech. Smoke
scores and plots do not establish detection quality; do not fine-tune a real
pretrained model on these fixtures for deployment.

## Audio and interval data contract

Put private audio and the JSONL manifest under ignored `data/local/vad/`:

```json
{"id":"clip-001","group_id":"session-001","split":"train","audio":"audio/clip-001.wav","speech_segments":[{"start":0.32,"end":1.84},{"start":2.10,"end":3.50}]}
```

Each row requires a unique `id`, nonempty `group_id`, a `train` / `validation` /
`test` split, a dataset-relative audio path, and `speech_segments`. Boundaries are
finite seconds from the original clip origin, sorted, nonoverlapping and within
audio duration. `[]` explicitly labels a pure nonspeech clip. An annotation must
cover **all speech** in the clip: unannotated time is treated as negative, not
unknown. Decide whether breaths, laughs and singing count as speech and document
that policy consistently; retain music, background noise and silence as negatives.

Audio must be finite normalized mono PCM (WAV/FLAC); resampling to the configured
8 kHz / 16 kHz rate is explicit. Stereo is rejected instead of silently downmixed.
Training duration bounds reject oversized clips without truncation. The default
real preset permits 30 seconds; explicitly configured training bounds permit up
to 300 seconds. This bounded guide eagerly caches frozen features in memory.

Split by original recording/session/speaker before generating clips. Groups and
identical decoded/resampled audio cannot cross splits; content hashes do not
detect near duplicates. Each split must contain both positive and negative frames.
Pure nonspeech recordings are valid individual rows. Labels use majority speech
occupancy in 32 ms windows; the final partial frame excludes padding from both
labels and loss weight. Label timing precision is therefore important.

## Adapt a real Silero model

Copy `configs/vad/silero.yaml` into ignored `configs/local/vad.yaml` and point it at
your consented, fully annotated corpus. The preset pins `silero-vad==6.2.3` and the
SHA-256 of its bundled JIT asset. Package/version or asset mismatches fail early;
native exports reload locally without looking up the original package asset.

```bash
python -m advai_notebooks.vad validate --config configs/local/vad.yaml
python -m advai_notebooks.vad run --config configs/local/vad.yaml
python -m advai_notebooks.vad run --config configs/local/vad.yaml \
  --resume outputs/vad/silero-16k/latest
```

Use the `output_dir/latest` from your own configuration. Training is single-device
fp32, with duration-weighted BCE and Adam/AdamW. The recurrent state resets per
recording, never carries into another row, and encoder/STFT parameters receive no
gradient. Resume is at epoch boundaries and rejects changed config, data, source,
environment or resolved device. Keep data/checkpoints on Drive in Colab.

The native 8 kHz and 16 kHz decoder paths are tested on CPU with synthetic inputs
for gradients and export parity. This is **not** validation of target-domain speech
quality, GPU fine-tuning, free-Colab compatibility or real-time latency. Evaluate
on real held-out recordings before choosing a deployment policy.

## Evaluation and validation-only calibration

Validation BCE selects the checkpoint, including the untrained epoch-zero baseline.
Test data is report-only. After selection, optional calibration searches the explicit
threshold grid on **validation interval-duration F1**. The exit threshold is the
entry threshold minus configured hysteresis, clamped at zero. Ties prefer the higher
entry threshold. Test annotations do not enter the search.

Reports provide duration-weighted frame Precision / Recall / F1, miss rate
(`FN / (TP + FN)`), false-alarm rate (`FP / (FP + TN)`), BCE and confusion counts in
seconds. `segment_*` metrics compare the actual postprocessed intervals against
reference intervals, with **no boundary tolerance collar**. Corpus metrics sum
counts, not per-clip percentages. Frame labels are quantized; interval metrics are
not interchangeable with frame metrics. Zero-denominator rates are reported as zero.

`export/` contains the selected weights **and calibrated postprocessing settings**.
`latest_export/` contains the final epoch at the original configured thresholds.
`calibration.json` records the search. The final report compares baseline and selected
weights under the same selected threshold policy; that policy is learned from the
selected model's validation data, not independently optimized for the baseline.

```bash
python -m advai_notebooks.vad evaluate --config configs/vad/smoke.yaml \
  --checkpoint outputs/vad/smoke/export --split test \
  --output outputs/vad/smoke/evaluation.json
python -m advai_notebooks.vad export --checkpoint outputs/vad/smoke/export \
  --output outputs/vad/smoke/copied-export
```

## Detect and segment offline

```bash
python -m advai_notebooks.vad detect --checkpoint outputs/vad/smoke/export \
  --audio data/sample/vad/audio/one.wav --output outputs/vad/smoke/detection.json
python -m advai_notebooks.vad segment --checkpoint outputs/vad/smoke/export \
  --audio data/sample/vad/audio/one.wav --output-dir outputs/vad/smoke/chunks \
  --max-chunk-duration 30
```

Detection returns 32 ms probabilities and speech intervals in seconds. Hysteresis
preserves speech between entry/exit thresholds, bridges configured short silences,
removes short speech regions, adds bounded boundary padding and merges touching
padded intervals. The saved policy uses the project's offline postprocessor, not
Silero's `get_speech_timestamps` implementation. These controls affect boundary
quality and must be validated together. Detection defaults to a 300-second bound;
the saved `max_inference_duration` can explicitly be raised to at most one hour.

For multiple files, `detect-batch --checkpoint ... --input inputs.json --output ...`
accepts a JSON list of audio paths or `{"audio":"path.wav"}` objects. All inference
commands can resolve a project Hub export with `--revision` and `--local-files-only`.

Segmentation writes only to a **new directory**. Each chunk retains `source_start`
and `source_end` in the original recording timeline; intervals longer than the
configured chunk limit are split without dropping samples. A limit is not a promise
of sentence boundaries. No detected speech produces an empty chunk list, not a
silent fallback to the entire recording. Raw audio and generated chunks remain
ignored by Git. Reports contain sensitive timing/path information: keep them private.

```python
from advai_notebooks.vad.model import VADModel
model = VADModel.load("outputs/vad/smoke/export", "cpu")
detection = model.detect("data/sample/vad/audio/one.wav")
print(detection["speech_segments"])
```

## Optional STT and TTS integration

For STT inference: long audio → VAD chunks → transcription → merge using source
offsets. For Whisper, choose a chunk limit no greater than its 30-second audio
budget, but splitting alone does not guarantee timestamped or word-level alignment.
The manifest deliberately contains **no transcript labels**. Do not assign an
entire recording's transcript to every chunk: align or re-annotate first if using
the chunks for supervised training. This guide does not automate transcript merging.

For TTS preparation, VAD can assist inspection of leading/trailing silence. Do not
automatically remove sentence-internal pauses or breaths; keep text/audio alignment
and prosody intact. Neither STT nor TTS requires VAD as a mandatory dependency.

## Native export, publication and licenses

Real exports include `silero_vad.jit`, project metadata and the upstream MIT notice.
The selected decoder weights are installed into the native JIT; the other sample-rate
branch remains unchanged. Raw `audio_forward` probabilities match the project wrapper
on tested inputs, but native callers must apply the metadata's calibration and
postprocessing policy separately. Tiny exports instead use `model.safetensors`.
Load only trusted JIT/checkpoint files.

`publish --export-dir ... --repo-id owner/model --dry-run` previews a whitelist-only
inference package. Raw recordings, manifests and optimizer state are not uploaded.
Publication is explicit; check consent and dataset licensing before publishing weights.

## Backend references

- [Silero VAD repository and license](https://github.com/snakers4/silero-vad)
- [Official decoder-tuning interface](https://github.com/snakers4/silero-vad/tree/master/tuning)
- [Pinned package release](https://pypi.org/project/silero-vad/6.2.3/)
