# Synthetic VAD fixtures

Eight deterministic 2-second mono PCM WAV files at 16 kHz, with four training,
two validation and two test rows. Two clips contain only a low-amplitude tone;
the others contain louder tone regions with artificial interval labels.
**None are human speech.** The labels test interval mechanics, not recognition.

These project-generated fixtures use the repository MIT license and contain no
personal recordings. Recreate exactly with `scripts/prepare_vad_fixtures.py`;
the generator refuses to overwrite differing files. Only the eight named WAVs
and this manifest/README are admitted by `.gitignore`.
