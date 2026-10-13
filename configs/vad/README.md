# VAD configurations

`smoke.yaml` uses a random small encoder/decoder and synthetic tone fixtures.
`silero.yaml` pins the Silero 6.2.3 package and JIT SHA-256 for decoder-only tuning;
point its manifest at your own fully annotated speech corpus. Supports 8/16 kHz,
single-device fp32, frozen features and validation-only threshold calibration.
See [English](../../docs/vad.md) / [简体中文](../../docs/vad.zh-CN.md).
