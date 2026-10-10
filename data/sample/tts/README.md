# Synthetic TTS fixtures

Eight 0.25-second mono 24 kHz PCM16 sine waves generated locally by
`python scripts/prepare_speech_fixtures.py`, with fixed frequencies and amplitude.
No third-party recordings or personal data are used. The project MIT license applies.
`speaker_id: synthetic` and text labels are placeholders, not a human voice corpus.
The 4/2/2 split tests pipeline mechanics; `audio/one.wav` is a training-only reference.
Do not fine-tune a real Qwen model on these tones or judge generated speech from them.
Use consented recordings under ignored `data/local/tts/`; see [TTS](../../../docs/tts.md).
