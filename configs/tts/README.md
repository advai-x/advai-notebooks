# TTS configurations

- `smoke.yaml`: synthetic text-conditioned GRU and tone fixtures, CPU fp32; not Qwen.
- `qwen3.yaml`: pinned Qwen3-TTS 12Hz 0.6B Base, single speaker, GPU fp32.

Copy into ignored `configs/local/`, replace manifest/output and training-only voice
reference paths, then validate. The real preset does not guarantee free-Colab fit.
See the [guide](../../docs/tts.md) for dependency isolation, consent and listening checks.
