# Scripts

Command-line entry points for repeatable data preparation, training, evaluation and export jobs.

`validate_notebooks.py` uses nbformat/nbclient to validate notebooks, optionally execute
them top-to-bottom and save outputs, and render HTML previews with nbconvert.

```bash
python scripts/validate_notebooks.py notebooks/vlm/01_finetune.ipynb
python scripts/validate_notebooks.py notebooks/vlm/01_finetune.ipynb \
  --execute --kernel python3 --html-dir outputs/notebook-previews
```

The chosen kernel must have the relevant task extras installed. Executing a tutorial
creates a fresh experiment under ignored `outputs/`; it does not upload anything.

`prepare_speech_fixtures.py` deterministically recreates the small checked-in sine-wave
WAV fixtures; it refuses to overwrite different audio. These tones are not real speech.
`build_speech_notebooks.py` scaffolds paired English/Chinese speech tutorials with
nbformat. Existing notebooks require explicit `--replace`, which clears saved outputs;
rerun `validate_notebooks.py --execute` after regenerating.
