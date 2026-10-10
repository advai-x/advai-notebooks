# Synthetic STT fixtures

Eight 0.25-second mono 16 kHz PCM16 sine waves generated locally by
`python scripts/prepare_speech_fixtures.py`, with fixed frequencies and amplitude.
No third-party recordings or personal data are used. The project MIT license applies.
The text labels are arbitrary color words: **these tones do not contain speech**.
They validate contracts and mechanics only, not WER/CER or model quality.
Each clip has its own group; the manifest provides 4/2/2 train/validation/test rows.
Real recordings belong under ignored `data/local/stt/`; see [STT](../../../docs/stt.md).
