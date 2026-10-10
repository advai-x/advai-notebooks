# STT configurations

- `smoke.yaml`: offline random native Whisper, synthetic tones, CPU fp32.
- `hf.yaml`: pinned multilingual Whisper-tiny, Chinese transcription, GPU fp32.

Copy the real preset into ignored `configs/local/`, replace dataset/output paths,
language and budgets, then validate. Whisper-base requires its own pinned revision.
See the [guide](../../docs/stt.md); do not mix speech and Transformers 5.x extras.
