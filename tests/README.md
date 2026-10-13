# Tests

Fast unit, contract and tiny CPU lifecycle tests; no expensive real-model downloads are needed.

```bash
python -m pytest tests/test_multimodal_contracts.py tests/test_multimodal_lifecycle.py tests/vlm tests/vla tests/vm -q
```

The multimodal tasks cover data/media leakage guards, normalization, padding/answer masks,
native export/reload, inference, exact interrupted epoch recovery, offline publication and task isolation.
GPU and long-running model-quality checks are outside the smoke suite.

Speech uses a separate environment because Qwen3-TTS pins Transformers 4.57.3,
while the multimodal suite uses 5.18.0:

```bash
python -m pip install -e '.[stt,tts,dev]'
python -m pytest tests/test_speech.py -q
```

The speech suite covers audio bounds/path/split/reference checks, corpus WER/CER,
CLI/inference, exact epoch recovery, native small-Qwen gradients and SDK/export
boundaries. Native codec and pretrained GPU quality are not established by these
CPU tests; boundary spies are explicitly labeled. Without the TTS extra, native
Qwen tests skip rather than download weights. Execute both tutorial languages with
`scripts/validate_notebooks.py` in a kernel with speech/notebooks extras installed.

VAD has no Transformers dependency. Run `python -m pytest tests/test_vad.py -q`
after installing `.[vad,dev,publish]`. Tests cover interval/audio contracts,
duration-weighted metrics, offline calibration, exact resume, safe chunk offsets,
actual bundled Silero decoder gradients and native JIT parity at 8/16 kHz.
Native tests use synthetic input and do not establish real speech quality.
