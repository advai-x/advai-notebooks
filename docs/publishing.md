# Publish models to Hugging Face

English | [简体中文](publishing.zh-CN.md)

The lifecycle is data → train/fine-tune → evaluate/select → native export → optional
publication → revision-pinned inference. All eleven tasks publish to independent
model repositories with the same publication arguments. Publication is explicit:
`run` never uploads automatically, and notebook examples default to offline previews.

## Installation and authentication

```bash
python -m pip install -e '.[embedding,ranker,decision,publish]'
hf auth login
```

Install the extra for your selected task instead when using another model family.
Alternatively, set `HF_TOKEN` to a credential with write access to the target repository.
Never put tokens in YAML, model cards, manifests or command-line arguments.
The Python API's `token` argument is passed only to the Hub client.

## Validate the package first

`--export-dir` points to an inference export, such as `output_dir/export` produced
by `run`. Publication filters inference files, creates a model card and actually
reloads the filtered package. Missing required files fail before upload.
Dry runs do not access the Hub or read login credentials.

```bash
python -m advai_notebooks.embedding publish \
  --export-dir outputs/embedding/smoke/export \
  --repo-id your-org/your-embedding --dry-run \
  --output outputs/publishing/embedding-plan.json
python -m advai_notebooks.ranker publish \
  --export-dir outputs/ranker/smoke/export \
  --repo-id your-org/your-ranker --dry-run \
  --output outputs/publishing/ranker-plan.json
python -m advai_notebooks.decision publish \
  --export-dir outputs/decision/smoke/export \
  --repo-id your-org/your-decision --dry-run \
  --output outputs/publishing/decision-plan.json
```

Previews include repository, visibility, branch, file sizes/SHA256 and the full
model card. They do not alter the original export. Packages exclude run directories,
raw data, training labels, candidates, optimizer/RNG state, logs and `.env` files.

| Task | Published assets | Loading/inference |
| --- | --- | --- |
| Embedding | Native SentenceTransformer components/weights/tokenizer and prefix metadata | `EmbeddingModel.load/encode`, or native SentenceTransformer |
| Ranker | Native CrossEncoder components/weights/tokenizer and prefix/raw-score metadata | `RankerModel.load/score`, or native CrossEncoder |
| Decision | Full backbone/head weights, encoder config, tokenizer, task definitions and calibration | `DecisionModel.load/system_one` |
| LLM Base | Full/adapter weights, tokenizer, generation config, base provenance | `BaseModel.load/generate` |
| LLM Instruct | Full/adapter weights, tokenizer/chat template, generation config, base provenance | `InstructModel.load/chat` |
| LLM PPO | Full/adapter actor, value head, tokenizer/template, generation config | `PPOModel.load/generate/chat` |
| LLM GRPO | Full/adapter actor, tokenizer/template, generation config | `GRPOModel.load/generate/chat` |
| Diffusion | Native full pipeline or LoRA/bundled base, component configs and noise scheduler | `DiffusionModel.load/generate` |
| VLM | Native LLaVA full or LoRA/bundled base, processor/tokenizer/template | `VLMModel.load/chat` |
| VLA | CLIP encoder/processor, action head, train-only statistics and physical semantics | `VLAModel.load/predict` |
| VM | Numerical dynamics safetensors, architecture, statistics and state/action semantics | `VMModel.load/predict/rollout` |

Decision questions, options and rules belong to its inference contract; `tasks.json`
is published. Generated `README.md` cards include loading examples and tiny-random
model caveats. Some tasks also include selected epochs and test summaries when a
matching run report is present. First VLM/VLA/VM cards explain backend scope and
reload entry points; supplement them with your data provenance, license and
evaluation before real publication. Full reports and sample predictions are excluded.
Standard published metadata strips authentication fields and replaces absolute
local base paths with `local`; weights and task definitions are unchanged.

## Upload explicitly

After choosing your repository and validating the model on domain data, remove
`--dry-run`:

```bash
python -m advai_notebooks.decision publish \
  --export-dir outputs/decision/domain-run/export \
  --repo-id your-org/your-decision \
  --model-card docs/model-cards/your-decision.md \
  --output outputs/publishing/decision-result.json
```

The custom card path above is a file you create, not a checked-in template.
Other tasks use the same arguments. New repositories default to private; `--public`
requests public visibility. Existing repository visibility must match the requested
setting; publication does not change it automatically. Use one model per repository.
Updates add/overwrite package files without deleting unrelated remote files.

`--revision` defaults to `main`; other values must identify an existing writable
branch. `--license` sets a generated-card license identifier; no license is inferred.
Custom Markdown supplied with `--model-card` is used verbatim, including its license,
training and evaluation statements, and overrides the generated card.

Use English Hub model cards. Recorded examples include
[Embedding](model-cards/wbs-embedding-demo.md),
[Ranker](model-cards/wbs-ranker-demo.md) and
[Decision](model-cards/wbs-decision-demo.md).
Input languages and tokenizer contents remain model-specific.

Packages include `publish_manifest.json` with schema version, task identity and
inference-file SHA256 values. Successful results contain repository URL, commit SHA
and commit URL; retain them for pinned deployment. Network/permission errors fail
explicitly and are not reported as successful publication.

## Load a pinned revision

Each task's `Model.load` accepts a local export or a project-published
`namespace/model`, plus `revision`, `token`, `cache_dir` and `local_files_only`.
Hub snapshots are downloaded before the task's native loader is used.

```python
from advai_notebooks.embedding.model import EmbeddingModel
from advai_notebooks.ranker.model import RankerModel
from advai_notebooks.decision.model import DecisionModel

embedding = EmbeddingModel.load("your-org/your-embedding", revision="EMBEDDING_COMMIT_SHA")
vectors = embedding.encode(["reset password"], role="query")

ranker = RankerModel.load("your-org/your-ranker", revision="RANKER_COMMIT_SHA")
scores = ranker.score([("reset password", "account login password reset")])

decision = DecisionModel.load("your-org/your-decision", revision="DECISION_COMMIT_SHA")
answers = decision.system_one("Customer message", task="your-task-id")
```

CLI `encode`/`rerank`/`predict`/`evaluate` support `--revision`, `--cache-dir` and
`--local-files-only`. Private repositories require login or `HF_TOKEN`.
Decision is project-owned architecture: install this project and use its loader.
Ordinary Transformers `AutoModel` will not recreate its head, input encoder or
typed decoder. Uploading files does not deploy a hosted inference service.

## Implementation and validation

Each task's `publish.py` owns file contracts/cards. `common/hub.py` handles neutral
filtering, packaging, upload and snapshot download, not task networks or metrics.
Repository/card/token settings stay outside training YAML to preserve task isolation.
VLM/VLA/VM additionally reuse neutral whitelist logic in `common/native_publication.py`.

The original retrieval/Decision checks cover real tiny native exports, offline CLI,
mock uploads/downloads, prefix/score/typed-output equality, task/calibration retention,
hashes, safe paths and existing-repository visibility. Later task suites cover their
own native exports and publication contracts. Offline tests never create Hub
repositories. Real uploads require a chosen target and authorized credentials.

Official references: [file uploads](https://huggingface.co/docs/huggingface_hub/guides/upload),
[model cards](https://huggingface.co/docs/huggingface_hub/guides/model-cards),
[snapshot downloads](https://huggingface.co/docs/huggingface_hub/guides/download).

## LLM full/adapter packages

Base/Instruct use the same arguments. Full models load natively; adapters need their
pinned base. Bundled bases are uploaded, external bases record immutable SHAs, and
chat-template `.jinja` files/multi-template directories are included. Offline adapter
preview uses cache only and fails when an external base is unavailable.
`export --merged` produces a standalone full model; see the [LLM guide](llm.md).

```bash
python -m advai_notebooks.llm.base publish \
  --export-dir outputs/llm/base/run/export \
  --repo-id your-org/your-llm-base --dry-run
python -m advai_notebooks.llm.instruct publish \
  --export-dir outputs/llm/instruct/run/export \
  --repo-id your-org/your-llm-instruct --dry-run
```

Recorded Base/Instruct/PPO/GRPO private demos have independent Datasets and English
cards with pinned data revisions. PPO/GRPO release verification reloaded local
packages and pinned Hub snapshots, comparing hashes, native/facade/CLI inference
and test reports. PPO includes `value_head.safetensors` for facade loading; native
causal-LM generation uses only the actor. See [Dataset mappings](datasets.md) and
the [historical RL release record](releases/20261008-rl-demo-v1.json).

## Diffusion packages

`python -m advai_notebooks.diffusion publish` uses the same arguments and private
default. Start with `--dry-run`. Its whitelist selects native Diffusers
`model_index.json`, component directories, safetensors, tokenizer and scheduler;
LoRA adds a frozen initial pipeline and native adapter metadata. Filtered packages
are reloaded before upload, excluding original images and generated PNGs.
The implementation did not upload a Diffusion model or Dataset.
See the [Diffusion guide](diffusion.md).

## Speech packages

STT/TTS use the same explicit publish flags in their separate speech environment:

```bash
python -m advai_notebooks.stt publish --export-dir outputs/stt/smoke/export \
  --repo-id your-org/your-stt-model --dry-run
python -m advai_notebooks.tts publish --export-dir outputs/tts/smoke/export \
  --repo-id your-org/your-tts-model --dry-run
```

Whisper exports include native weights, tokenizer and feature extraction settings.
Qwen TTS exports additionally include the speech codec and fixed speaker embedding;
they retain the Base format. Whitelists exclude raw recordings and training state,
but voice embeddings are derived biometric data and require explicit speaker
permission to publish. No STT/TTS model or Dataset was uploaded by this implementation.
See [STT](stt.md) and [TTS](tts.md) for scopes and third-party licensing requirements.
