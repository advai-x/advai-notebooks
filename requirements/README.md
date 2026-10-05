# Requirements

Keep the shared environment in `base.txt` and add task-specific dependency files as implementations are introduced.

`decision.txt` installs native Transformers/PyTorch/safetensors/NumPy dependencies. The preferred project install
is `python -m pip install -e '.[decision,notebooks,dev]'`. Install a matching
PyTorch build for the target hardware before the project. Each experiment records
the exact resolved Python/package versions in `environment.json`.

`embedding.txt` and `ranker.txt` pin SentenceTransformers to 5.7.0, matching the
tested preprocessing/export/loss APIs. Install with
`python -m pip install -e '.[embedding,ranker,notebooks,dev]'` for both workflows.
The tested CPU environment used Python 3.12, PyTorch 2.14.1 and Transformers
5.18.0; environment records capture exact versions for each run.
