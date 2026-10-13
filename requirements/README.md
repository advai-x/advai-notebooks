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

`stt.txt` and `tts.txt` use a **separate speech environment**. Both pin Transformers
4.57.3; TTS pins qwen-tts 0.1.1 (which requires Accelerate 1.12.0). Do not combine
them with task extras pinning Transformers 5.18.0 / Accelerate 1.15.0. STT/TTS may
be installed together with `python -m pip install -e '.[stt,tts,notebooks,dev]'`.
Use matching PyTorch/torchaudio builds and record resolved versions with each run.

`vad.txt` pins Silero VAD 6.2.3 and uses SoundFile/SciPy for audio I/O. VAD does not
require Transformers or TorchAudio; install with `.[vad,notebooks,dev,publish]`,
or add `vad` to an existing compatible speech environment.
