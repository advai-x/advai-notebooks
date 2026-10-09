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
