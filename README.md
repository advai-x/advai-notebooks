<p align="center">
  <img src="docs/assets/readme-hero.png" alt="advai-notebooks — practical, reproducible model training with notebooks" width="100%">
</p>

<h1 align="center">advai-notebooks</h1>

<p align="center">
  English | <a href="README.zh-CN.md">简体中文</a> | <a href="https://advai-x.github.io/advai-notebooks/en/">Documentation</a>
</p>

<p align="center">
  Practical, reproducible training guides for modern AI models.<br>
  From a tiny offline smoke run to a configurable experiment on your own data.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-training-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch training"></a>
  <a href="notebooks/"><img src="https://img.shields.io/badge/Google%20Colab-ready-F9AB00?logo=googlecolab&amp;logoColor=white" alt="Google Colab ready"></a>
  <a href="docs/publishing.md"><img src="https://img.shields.io/badge/Hugging%20Face-export%20ready-FFD21E" alt="Hugging Face export ready"></a>
  <a href="#project-status"><img src="https://img.shields.io/badge/status-active%20development-blue" alt="Status: active development"></a>
</p>

advai-notebooks is a collection of executable notebooks, YAML configurations and
reusable Python modules for learning and applying mainstream model-training workflows.
Every guide is designed to show the complete lifecycle:

```text
validate data → configure model → train → evaluate → resume → export → reload
```

The notebooks are intentionally thin: they explain and orchestrate the same tested
implementation used by the command-line interface. You can learn interactively in
Google Colab, then move the same configuration to a local machine or GPU environment.

> [!IMPORTANT]
> The checked-in smoke configurations use tiny random models and synthetic data.
> They verify that the pipeline works end to end; their metrics and generated outputs
> do **not** represent real model quality.

## Choose a training guide

| Track | What you will train | Supported methods | Start here |
| --- | --- | --- | --- |
| Embedding | Dense text retrieval encoder | MNRL, Cached MNRL, hard-negative preparation | [Notebook](notebooks/embedding/01_finetune.ipynb) · [Guide](notebooks/embedding/README.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/embedding/01_finetune.ipynb) |
| Ranker | CrossEncoder reranker | Pointwise BCE, ranking evaluation | [Notebook](notebooks/ranker/01_finetune.ipynb) · [Guide](notebooks/ranker/README.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/ranker/01_finetune.ipynb) |
| Decision | Typed choice, score and yes/no model | Supervised CE, ordinal loss, optional RLCD, calibration | [Notebook](notebooks/decision/01_finetune.ipynb) · [Guide](notebooks/decision/README.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/decision/01_finetune.ipynb) |
| LLM Base | Decoder language model | Continued pre-training, full fine-tuning, LoRA | [Notebook](notebooks/llm/base/01_finetune.ipynb) · [Guide](notebooks/llm/base/README.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/base/01_finetune.ipynb) |
| LLM Instruct | Instruction-following assistant | Assistant-only SFT, full fine-tuning, LoRA | [Notebook](notebooks/llm/instruct/01_finetune.ipynb) · [Guide](notebooks/llm/instruct/README.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/llm/instruct/01_finetune.ipynb) |
| LLM PPO | Language-model policy with value head | GAE, clipped policy/value objectives, full/LoRA | [RL guide](docs/rl.md) · [Configs](configs/llm/ppo/) |
| LLM GRPO | Language-model policy without a critic | Group-relative advantages, clipping, reference KL, full/LoRA | [RL guide](docs/rl.md) · [Configs](configs/llm/grpo/) |
| Diffusion | Latent text-to-image model | Full UNet fine-tuning, attention LoRA | [Notebook](notebooks/diffusion/01_finetune.ipynb) · [Guide](docs/diffusion.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/diffusion/01_finetune.ipynb) |
| VLM | Native LLaVA single-image question answering | Answer-only full / language LoRA fine-tuning | [Notebook](notebooks/vlm/01_finetune.ipynb) · [Guide](docs/vlm.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vlm/01_finetune.ipynb) |
| VLA | CLIP + language + state → continuous action chunks | Behavior-cloning baseline; frozen or full encoder updates | [Notebook](notebooks/vla/01_finetune.ipynb) · [Guide](docs/vla.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vla/01_finetune.ipynb) |
| VM | Action-conditioned numerical World Model | Residual dynamics, native checkpoint fine-tuning, recursive rollouts | [Notebook](notebooks/vm/01_finetune.ipynb) · [Guide](docs/vm.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vm/01_finetune.ipynb) |
| STT | Speech-to-text / ASR with native Whisper | Full supervised fine-tuning, WER/CER | [Notebook](notebooks/stt/01_finetune.ipynb) · [Guide](docs/stt.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/stt/01_finetune.ipynb) |
| VAD | Speech intervals with Silero VAD | Frozen encoder, decoder fine-tuning, validation-only calibration | [Notebook](notebooks/vad/01_finetune.ipynb) · [Guide](docs/vad.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/vad/01_finetune.ipynb) |
| TTS | Qwen3-TTS Base text-to-speech | Full single-speaker fine-tuning, non-streaming synthesis | [Notebook](notebooks/tts/01_finetune.ipynb) · [Guide](docs/tts.md) · [Open in Colab](https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/notebooks/tts/01_finetune.ipynb) |

VLM, VLA and VM are independent model families alongside `llm/`. In this repository,
`vm/` means **World Model**, rather than a vision-only or video-only model.
These first backends have narrow, explicit scope: VLM does not implement video/Qwen-VL,
VLA does not load OpenVLA checkpoints, and VM does not generate video. VLA/VM are
trainable baselines, not demonstrated robotics or planning capabilities.

`stt/` and `tts/` are independent speech tracks at the same level. Speech extras
require a **separate environment** (Transformers 4.57.3, versus 5.18.0 in several
other tracks). TTS smoke uses a synthetic GRU, not Qwen; native Qwen fine-tuning
requires consented speech data and target-hardware validation. See the speech guides.

`vad/` is a peer speech track for offline speech-interval detection and segmentation.
It has no Transformers dependency. Its smoke backend is random, not pretrained
Silero; the real backend tunes only the selected decoder. See the
[detection and segmentation tutorial](notebooks/vad/02_detect_and_segment.ipynb).

There is also an optional
[retrieval comparison notebook](notebooks/workflows/01_retrieval_comparison.ipynb)
for evaluating embedding retrieval and reranking together. The two models remain
independently trainable.

## Quick start

### Option A — Google Colab

Open a notebook from the [training directory](notebooks/README.md), then prepare
the repository in the Colab runtime:

```python
%cd /content
!test -d advai-notebooks || git clone --depth 1 https://github.com/advai-x/advai-notebooks.git
%cd /content/advai-notebooks
%pip install -e ".[embedding,notebooks]"
```

Replace `embedding` with the extra for your chosen track:
`ranker`, `decision`, `llm-base`, `llm-instruct`, `llm-ppo`,
`llm-grpo`, `diffusion`, `vlm`, `vla`, `vm`, `vad`, `stt` or `tts`.
For STT/TTS, use a fresh speech runtime; do not mix incompatible task extras.

For real training, keep datasets and checkpoints in persistent storage such as
Google Drive. A disconnected Colab runtime deletes files stored only under
`/content`.

### Option B — local smoke run

```bash
git clone https://github.com/advai-x/advai-notebooks.git
cd advai-notebooks

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[embedding,notebooks,dev]'

python -m advai_notebooks.embedding validate \
  --config configs/embedding/smoke.yaml
python -m advai_notebooks.embedding run \
  --config configs/embedding/smoke.yaml
```

The smoke run works offline after dependencies are installed and writes its artifacts
under `outputs/`, which is ignored by Git.

## From smoke test to a real experiment

1. Pick the closest guide and run its smoke configuration unchanged.
2. Copy the track's real-model preset, such as `hf.yaml`, `bge.yaml`, `lora.yaml`,
   `qwen3.yaml` or `silero.yaml`.
3. Pin a model ID and revision, then point the config at your own grouped data splits.
4. Validate the smoke config first. Model-aware validation of a real config loads its
   weights and may require the target hardware and memory budget.
5. Train into a fresh output directory and keep the configuration with the run.
6. Evaluate on held-out data, inspect the report, then export or publish deliberately.

Example:

```bash
cp configs/llm/instruct/hf.yaml configs/llm/instruct/my_experiment.yaml

python -m advai_notebooks.llm.instruct validate \
  --config configs/llm/instruct/my_experiment.yaml
python -m advai_notebooks.llm.instruct run \
  --config configs/llm/instruct/my_experiment.yaml
```

Most trainers support epoch-boundary recovery:

```bash
python -m advai_notebooks.embedding run \
  --config configs/embedding/my_experiment.yaml \
  --resume outputs/embedding/my_experiment/latest
```

Resume is intentionally strict: configuration, data, source fingerprints and relevant
package versions must still match the original run.

## What makes these guides different?

- **End-to-end guides.** Each implemented guide covers validation, training, evaluation,
  checkpointing, export and reload.
- **Notebook and CLI parity.** Interactive notebooks call the same modules as automated
  command-line runs.
- **Offline-first smoke tests.** Tiny local models and synthetic fixtures expose setup
  and data-contract problems before expensive training.
- **Config-driven experiments.** Model IDs, data paths, runtime choices and
  hyperparameters live in versionable YAML files.
- **Task isolation.** Embedding, ranker, decision, LLM, diffusion, VLM, VLA and VM own
  their model-specific data, losses and metrics, as do VAD, STT and TTS.
- **Leakage-aware validation.** The data loaders validate split boundaries, groups and
  task-specific label contracts.
- **Reproducible artifacts.** Runs record configuration, environment, metrics,
  checkpoints and model metadata.
- **Deliberate publishing.** Hugging Face publication supports dry-run previews and does
  not upload training data automatically.

## Command-line overview

All task CLIs follow the same general shape:

```text
python -m advai_notebooks.<task> validate --config <config.yaml>
python -m advai_notebooks.<task> run      --config <config.yaml>
```

Common entry points:

```bash
python -m advai_notebooks.embedding run   --config configs/embedding/smoke.yaml
python -m advai_notebooks.ranker run      --config configs/ranker/smoke.yaml
python -m advai_notebooks.decision run    --config configs/decision/smoke.yaml
python -m advai_notebooks.llm.base run    --config configs/llm/base/smoke.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/smoke.yaml
python -m advai_notebooks.llm.ppo run     --config configs/llm/ppo/smoke.yaml
python -m advai_notebooks.llm.grpo run    --config configs/llm/grpo/smoke.yaml
python -m advai_notebooks.diffusion run   --config configs/diffusion/smoke.yaml
python -m advai_notebooks.vlm run         --config configs/vlm/smoke.yaml
python -m advai_notebooks.vla run         --config configs/vla/smoke.yaml
python -m advai_notebooks.vm run          --config configs/vm/smoke.yaml
python -m advai_notebooks.stt run         --config configs/stt/smoke.yaml
python -m advai_notebooks.vad run         --config configs/vad/smoke.yaml
python -m advai_notebooks.tts run         --config configs/tts/smoke.yaml
```

Data preparation and combined retrieval evaluation are separate, optional stages:

```bash
python -m advai_notebooks.data_preparation prepare \
  --config configs/data_preparation/smoke.yaml

python -m advai_notebooks.workflows compare \
  --config configs/workflows/smoke.yaml \
  --embedding-run outputs/embedding/smoke \
  --ranker-run outputs/ranker/smoke \
  --output outputs/retrieval/comparison.json
```

## Repository layout

```text
advai-notebooks/
├── notebooks/                  # Human-readable, executable walkthroughs
│   ├── embedding/
│   ├── ranker/
│   ├── decision/
│   ├── llm/{base,instruct}/
│   ├── vlm/                    # Native LLaVA image QA
│   ├── vla/                    # Continuous-action behavior cloning
│   ├── vm/                     # Numerical world dynamics
│   ├── diffusion/
│   ├── vad/ stt/ tts/           # Speech detection, recognition and synthesis
│   └── workflows/
├── configs/                    # Smoke, real-model and LoRA experiment presets
├── src/advai_notebooks/        # Reusable training and inference implementations
│   ├── embedding/
│   ├── ranker/
│   ├── decision/
│   ├── llm/{base,instruct,ppo,grpo}/
│   ├── vlm/
│   ├── vla/
│   ├── vm/
│   ├── diffusion/
│   ├── vad/ stt/ tts/
│   ├── data_preparation/
│   ├── workflows/
│   └── common/
├── data/sample/                # Small versioned fixtures and schema examples
├── docs/                       # Architecture and task-specific references
├── site/                       # Bilingual documentation site and content adapter
├── tests/                      # Unit, contract and lifecycle regression tests
└── outputs/                    # Local generated artifacts; ignored by Git
```

## Documentation

The [documentation website](https://advai-x.github.io/advai-notebooks/en/) provides
search, grouped navigation and matching Chinese pages. It becomes available after
GitHub Pages is enabled and the documentation workflow is deployed. For local
preview and setup, see the [site guide](site/README.md). The files below remain
the source of truth for guide content.

- [Architecture and extension map](docs/architecture.md)
- [Dataset contracts and Hub datasets](docs/datasets.md)
- [Embedding guide](notebooks/embedding/README.md)
- [Ranker guide](notebooks/ranker/README.md)
- [Decision-model guide](notebooks/decision/README.md)
- [LLM base and instruct guide](docs/llm.md)
- [Native MoE CPT/SFT: OLMoE & Qwen3-MoE](docs/llm-moe.md)
- [PPO and GRPO guide](docs/rl.md)
- [VLM fine-tuning guide](docs/vlm.md)
- [VLA action-training guide](docs/vla.md)
- [VM / World Model guide](docs/vm.md)
- [Diffusion guide](docs/diffusion.md)
- [STT / Whisper guide](docs/stt.md)
- [VAD / Silero guide](docs/vad.md)
- [TTS / Qwen3-TTS guide](docs/tts.md)
- [Hugging Face publication guide](docs/publishing.md)
- [Retrieval data contract](data/sample/retrieval/README.md)

## Data, secrets and generated files

- Keep large or sensitive datasets outside Git; `data/local/` is ignored.
- Keep tokens in environment variables or Colab Secrets, never in notebooks or YAML.
- Model weights, checkpoints and generated outputs under `outputs/` are ignored.
- Commit small samples, schemas and manifests only when redistribution is permitted.
- Review model and dataset licenses before downloading, training or publishing.

## Development

Install the extras for the area you are changing together with `dev`:

```bash
python -m pip install -e '.[embedding,dev]'
python -m pytest tests/embedding
python -m ruff check src tests
```

The complete test suite spans several optional dependency groups. Contributors can run
the focused task suite locally and should describe any hardware-dependent validation
performed with a real checkpoint.

## Contributing

Issues, corrections and new training guides are welcome. For a focused contribution:

1. Open an issue describing the model family, training objective and expected output.
2. Keep the notebook instructional and move reusable behavior into
   `src/advai_notebooks/`.
3. Add a tiny offline fixture or smoke configuration whenever practical.
4. Add tests for data contracts, metrics, checkpoint recovery and export/reload behavior.
5. Update the relevant guide with supported hardware and known limitations.
6. Do not commit private datasets, access tokens or large model artifacts.

Use English for the default guides, notebook explanations, comments and reader-facing messages.
Keep Chinese companions in `*.zh-CN.md` and `*.zh-CN.ipynb`, with language-switch
links between versions, and update both when a documented contract changes.
Keep model-required prompts and dataset examples in their original language when
that language is part of the training or inference contract.

## Project status

This repository is at version `0.1.0` and is under active development. The smoke
workflows are intended to be reproducible; real-model presets are starting points and
must be validated on your own data, hardware and quality criteria. Multi-GPU support
and production-scale performance vary by task and are not implied by the examples.

## License

Except where otherwise noted, the original code, notebooks, configurations and
documentation in this repository are licensed under the [MIT License](LICENSE).
When copying or redistributing the software or substantial portions of it, retain
the copyright and permission notices in that file.

Third-party dependencies, model weights and datasets remain subject to their own
licenses and terms. Training with this project does not automatically license the
resulting checkpoints or adapters under MIT; review the applicable permissions and
restrictions before using or publishing them.
