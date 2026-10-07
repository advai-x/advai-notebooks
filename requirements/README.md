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

`publish.txt` adds the Hugging Face Hub client (`huggingface-hub>=1,<2`).
Install with a task extra, for example `python -m pip install -e '.[decision,publish]'`.
See [the publication guide](../docs/publishing.md) for authentication and upload commands.

`llm-base.txt` and `llm-instruct.txt` install independent task extras, each using
Transformers 5.18.0, PEFT 0.18.1 and Accelerate 1.15.0. Install both with
`python -m pip install -e '.[llm-base,llm-instruct,notebooks,dev]'`.
The first version supports full/LoRA single-device training; QLoRA and distributed
training require later implementations, not additional flags alone.

`llm-ppo.txt` and `llm-grpo.txt` install the independent online RL task extras.
These first RL implementations use single-device fp32, full/LoRA and PyTorch training loops;
see [the RL guide](../docs/rl.md) for algorithm and checkpoint contracts.
