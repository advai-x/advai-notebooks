# advai-notebooks

Notebooks and reusable Python code for model fine-tuning and evaluation in local and Google Colab environments.

## Layout

```text
notebooks/
  _shared/                  # Shared Colab setup and utilities
  embedding/                # Embedding fine-tuning
  ranker/                   # Reranker fine-tuning
  decision/                 # Decision-model fine-tuning and evaluation
  llm/
    base/                   # Base-model continued pre-training
    instruct/               # Instruction tuning

configs/                    # Independent task, preparation and workflow settings
src/advai_notebooks/        # Task modules + neutral common utilities
  data_preparation/         # Optional offline mining / candidate preparation
  workflows/                # Optional combined-model evaluation
tests/                      # Unit and regression tests
data/                       # Local/sample data; large data is not committed
outputs/                    # Generated checkpoints and reports; not committed
requirements/               # Environment-specific dependencies
```

Notebook files orchestrate experiments and present results. Data processing, trainers, metrics and SDK integrations belong in `src/advai_notebooks` so they can be tested and reused across Colab sessions.

## Embedding and ranker fine-tuning v1

The [embedding notebook](notebooks/embedding/01_finetune.ipynb) and
[ranker notebook](notebooks/ranker/01_finetune.ipynb) now contain executable
end-to-end workflows. They share neutral corpus/query/qrels contracts and optimizer execution. Each owns
its Recall/MRR/nDCG metrics. Sampling, losses, data validation and training callbacks
live in each task separately.
Embedding uses MNRL or Cached MNRL; ranker uses a single-logit CrossEncoder with
binary cross-entropy. Each reads its own prepared training samples. Ranker reads
external validation/test candidates and never loads an embedding model during
training, evaluation or inference. Models and hardware remain YAML settings.

```bash
python -m pip install -e '.[embedding,ranker,notebooks,dev]'
python -m advai_notebooks.embedding validate --config configs/embedding/smoke.yaml
python -m advai_notebooks.embedding run --config configs/embedding/smoke.yaml
python -m advai_notebooks.ranker run --config configs/ranker/smoke.yaml
python -m advai_notebooks.workflows compare --config configs/workflows/smoke.yaml \
  --embedding-run outputs/embedding/smoke --ranker-run outputs/ranker/smoke \
  --output outputs/retrieval/comparison.json
```

Smoke runs use offline random tiny BERT models and synthetic data. Read the
[embedding guide](notebooks/embedding/README.md), [ranker guide](notebooks/ranker/README.md)
and [shared data contract](data/sample/retrieval/README.md) before using domain data.
The BGE example configs load real checkpoints and require your own data; they
have not been trained or benchmarked here.

## Boundaries and fine-grained changes

See [the architecture and adjustment map](docs/architecture.md) for module
boundaries, file contracts and migration details.

All seven tasks own `backbones.py` and `heads.py` for their network components;
`model.py` composes them and implements loading, saving and public inference.
Embedding and ranker also each have `config.py`, `data.py`, `sampling.py`, `losses.py`,
`train.py`, `metrics.py`, `evaluate.py`, `optimizer.py`, `pipeline.py` and `cli.py`. Change the corresponding
module to adjust one task without editing the other. `model.py` also exposes the
independent inference adapter and native checkpoint export.

`common/` provides task-neutral configuration, runtime, data contracts, artifacts,
CLI utilities and optimizer execution. Neither trainer imports the other task.
Negative mining belongs to `data_preparation/`; combined evaluation belongs to
`workflows/`. The two trainers may share physical data files, but each configuration
can point to a different corpus, query set, qrels and prepared samples.

The checked-in tiny samples and full-corpus candidates allow both smoke trainers
to run independently without preparing data first. To regenerate samples:

```bash
python -m advai_notebooks.data_preparation prepare --config configs/data_preparation/smoke.yaml
```

The explicit preparation step writes `embedding_samples.json`,
`ranker_samples.json`, `candidates.json` and a manifest. Set the matching
`data.training_samples` and ranker's `data.candidates` paths in your training
configs. A preparation config may use judged negatives with no model, or an
optional embedding model for hard negatives/candidates. External retrievers can
write the same candidate format.

`configs/data_preparation/bge.yaml` prepares real-domain artifacts with a chosen
encoder. `configs/workflows/bge.yaml` configures combined evaluation separately.
[The combined notebook](notebooks/workflows/01_retrieval_comparison.ipynb) is
optional; the ranker notebook now trains ranker only.

Migration from the earlier combined implementation:

- Move `training.negatives_per_query` to preparation's `mining.negatives_per_query`.
- Replace ranker's `retrieval` section with `data.training_samples` and `data.candidates`.
- Move encoder/prefix settings for mining to the preparation config.
- Replace `embedding mine` with `data_preparation prepare`; replace `ranker compare`
  with `workflows compare`. Python inference APIs and existing exports remain compatible.

## Decision fine-tuning v1

The [native decision pipeline](notebooks/decision/README.md) supports:
typed `choice`, `score` and `noul` tasks, supervised CE/ordinal training and optional RLCD experiments, held-out calibration,
evaluation, resumable epoch checkpoints and SDK-compatible exports. Decision has its
own sampler and optimizer; only neutral runtime/checkpoint utilities are shared.

```bash
python -m pip install -e '.[decision,notebooks,dev]'
python -m advai_notebooks.decision validate --config configs/decision/smoke.yaml
python -m advai_notebooks.decision run --config configs/decision/smoke.yaml
```

The smoke configuration uses a random tiny encoder and synthetic data. It checks
pipeline correctness; its scores do not measure domain quality. LLM Base and
Instruct fine-tuning workflows are described below.

## Training lifecycle and independent adjustment

After exporting, all three tasks support an explicit `publish` command with
`--export-dir`, `--repo-id`, `--dry-run` and optional `--public` / `--model-card`.
Inference files, task-specific metadata and a model card are uploaded to a Hugging Face
model repository; training data and resumable state stay local. All three `Model.load`
methods accept Hub repository IDs and a versioned `revision` as well as local paths.
See [the publication guide](docs/publishing.md) for authentication, upload previews,
private/public repositories and loading a pinned commit for inference.

Published demo models are associated with seven independent private demo datasets. Their model
cards identify the dataset IDs and pinned data revisions; dataset repositories
preserve the original source files and explicit loading subsets/splits.
See [the dataset guide](docs/datasets.md) for mappings and verified loading examples.

All three tasks support `run --resume <output_dir>/latest` at completed epoch
boundaries. Checkpoints atomically store current/best weights, optimizer, AMP and
Python/NumPy/Torch/device random states. Interrupted native weight exports can be
repaired from training state. Resume uses the original output directory and unchanged
configuration (including the planned epoch count), data, package versions and
relevant source code. An inference-only export is not a resumable checkpoint.

```bash
python -m advai_notebooks.embedding run --config configs/embedding/smoke.yaml --resume outputs/embedding/smoke/latest
python -m advai_notebooks.ranker run --config configs/ranker/smoke.yaml --resume outputs/ranker/smoke/latest
python -m advai_notebooks.decision run --config configs/decision/smoke.yaml --resume outputs/decision/smoke/latest
```

Each task selects `training.optimizer: adamw` or `adam` through its own
`optimizer.py`; custom parameter groups belong there. Embedding/ranker inject
that factory into the shared execution loop. Decision retains its own typed-option training loop
and separate encoder/head learning rates.

Decision defaults to `evaluation.selection_metric: nll` and
`evaluation.include_baseline: true`. `brier`, `ece` (lower is better) and
`accuracy` (higher is better) are also supported. Epoch zero can win when training
does not improve validation quality. Set `include_baseline: false` to select
among trained epochs only. `calibration.enabled: false` skips temperature fitting
and permits train/validation/test data without a calibration split; calibrated
metrics are then null and the export uses temperature 1.

Source fingerprints follow each task's explicit internal import dependencies.
Unrelated retrieval code changes no longer invalidate decision resume; changes to
shared runtime/checkpoint behavior still do. Older exports remain valid for
inference; training states from earlier implementations fail the strict resume
check. All recovery equivalence checks here use tiny models on CPU/fp32.

## Native decision model

Decision uses a Hugging Face base encoder and a task-type embedding + shared MLP
option scorer. Each state/question/option pair is encoded separately; softmax
across a question's options produces its distribution. Choice selects an option,
score returns an expected ordered level, and noul returns the true probability.
The head starts randomly when loading a base encoder and requires domain training.

Use `configs/decision/hf.yaml` for a multilingual base encoder or set `model.id`
to a compatible local/Hub text encoder. Pooling, head width/dropout, encoder freezing
and separate learning rates are configurable. Native pairing, ragged batching,
supervised ordinal loss, optional Brier/ordinal RLCD reward and temperature fitting
require no external decision SDK. The public `DecisionModel.system_one`
API accepts a state and named task (using exported task definitions), or a list
of label-free `DecisionRequest` objects. Choice, score and noul all return
`probabilities` and `confidence`; noul is the true probability in [0, 1].
`DecisionModel.load/save/system_one`
provides the public model facade and implements initialization/loading/saving
directly in `model.py`, without a separate provider or Bundle layer. Data validation
is in `data.py`, with
evaluation/calibration orchestration in `evaluate.py`. Component contracts and
extension boundaries are documented in [architecture.md](docs/architecture.md).
See [the notebook](notebooks/decision/01_finetune.ipynb) and
[decision guide](notebooks/decision/README.md). The earlier Laya provider and
SDK checkpoint format have been removed; old weights require retraining for the
new architecture. Task definitions, labels and typed inference responses retain
their existing contract.

## Task-owned metrics

Each task independently owns its `metrics.py`: formulas, record validation,
metric selection and improvement comparison. No shared metrics implementation
or cross-task metrics imports remain. Embedding/ranker pass their own selection
callbacks to the neutral training loop; the combined retrieval workflow calls
the metrics belonging to each stage.
Task metric reports retain their existing value fields and add `task`,
`metric_schema_version: 1` and `metric_records` with `name`, `value`, `n`, `group`
and explicit `direction`. Checkpoint selection reads the overall metric record.
Ranking sample counts refer to queries; Decision ordinal-error counts refer only
to score decisions. See the [architecture guide](docs/architecture.md) for the contract.


## LLM Base and Instruct fine-tuning v1

[Base notebook](notebooks/llm/base/01_finetune.ipynb) implements continued pre-training;
[Instruct notebook](notebooks/llm/instruct/01_finetune.ipynb) implements assistant-only SFT.
Both independently own data, loss, metrics, evaluation, training assembly, model facade,
CLI and publication. Transformers supplies the original decoder/LM head and Trainer;
PEFT supplies configurable LoRA. Native weight tying and forward behavior are retained.

```bash
python -m pip install -e '.[llm-base,llm-instruct,notebooks,dev]'
python -m advai_notebooks.llm.base validate --config configs/llm/base/smoke.yaml
python -m advai_notebooks.llm.base run --config configs/llm/base/smoke.yaml
python -m advai_notebooks.llm.instruct run --config configs/llm/instruct/smoke.yaml
```

Use each task's `lora.yaml` for adapter training. `BaseModel.generate` returns continuations;
`InstructModel.chat` returns an assistant response using the exported chat template.
Base full/merged exports can optionally initialize Instruct through native files,
without importing Base task code. Validation token-weighted NLL selects checkpoints,
including epoch zero; test is report-only. Instruct also supports greedy exact-match
and JSON-validity evaluation. Metrics remain task-owned.

Full models, portable adapters and merged full exports can be reloaded and published.
Publication defaults to private and offline dry-run previews; no LLM repository is
uploaded automatically. Complete native Trainer checkpoints support strict epoch
recovery, while inference exports exclude optimizer/RNG state.
See [the LLM guide](docs/llm.md) for schemas, configuration, CLI and current limits.
The offline tiny decoder and synthetic samples verify functionality, not domain quality.
QLoRA, packing, distributed and preference training are not implemented in v1.

## LLM PPO and GRPO

Two independent online RL tasks now live in `src/advai_notebooks/llm/ppo/` and
`src/advai_notebooks/llm/grpo/`. PPO trains a shared-backbone value head with GAE
and clipped policy/value objectives. GRPO uses multiple completions per prompt,
group-relative advantages, clipping and reference KL, without a critic.
Both own their reward functions, metrics, trainers, model facade and CLI.
Instruct remains assistant-only SFT.

```bash
python -m pip install -e '.[llm-ppo,llm-grpo,dev]'
python -m advai_notebooks.llm.ppo run --config configs/llm/ppo/smoke.yaml
python -m advai_notebooks.llm.grpo run --config configs/llm/grpo/smoke.yaml
```

Each supports full/LoRA, deterministic held-out reward evaluation, epoch recovery,
native HF actor exports, generate/chat and explicit offline publication previews.
The first version uses single-device fp32 and rule/custom terminal rewards.
Tiny fixtures validate behavior; they do not establish model quality.
See [the RL guide](docs/rl.md) for algorithm details and limits.

PPO and GRPO trained demo models and their separate datasets are published privately
under `widebluesky/wbs-llm-ppo-demo` and `widebluesky/wbs-llm-grpo-demo` (`demo-v1`).
English cards identify the initial SFT policy and pinned dataset revisions. Fresh Hub
reload reproduced native/facade/CLI inference and held-out evaluation. Both models
have zero held-out exact match; these tiny synthetic releases validate the lifecycle
only. See [the release record](docs/releases/20261008-rl-demo-v1.json).

## Diffusion image fine-tuning

`diffusion/` is an independent peer of Embedding, Ranker, Decision and LLM.
The first backend uses native Diffusers Stable Diffusion components. Full updates
only the UNet; LoRA updates its attention adapters. The VAE and text encoder stay
frozen. Model loading, saving and image generation live in `DiffusionModel`.

```bash
python -m pip install -e '.[diffusion,notebooks,dev]'
python -m advai_notebooks.diffusion validate --config configs/diffusion/smoke.yaml
python -m advai_notebooks.diffusion run --config configs/diffusion/smoke.yaml
python -m advai_notebooks.diffusion run --config configs/diffusion/lora.yaml
```

Each task owns its data, noise sampling, denoising losses, optimizer, metrics,
validation selection, checkpoints, CLI and publication contract. Offline tiny
full/LoRA exports and native reload are supported, including portable bundled-base
adapters and merged full pipelines. Fixed-noise validation includes epoch zero;
held-out test is report-only. Generated examples use fixed seeds.
Single-device fp32 is supported; CUDA/MPS and real pretrained image quality need
hardware-specific validation. SDXL/DiT/video and AMP are outside the first backend.
See [the guide](docs/diffusion.md) and [the notebook](notebooks/diffusion/01_finetune.ipynb).
