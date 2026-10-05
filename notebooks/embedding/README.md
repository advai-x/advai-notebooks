# Embedding fine-tuning v1

Open [01_finetune.ipynb](01_finetune.ipynb) for the complete offline smoke workflow.
The implementation lives in `src/advai_notebooks/embedding`; the notebook uses
the same pipeline as the CLI. Python 3.10+ is required; Python 3.12 was tested.

```bash
python -m pip install -e '.[embedding,notebooks,dev]'
python -m advai_notebooks.embedding validate --config configs/embedding/smoke.yaml
python -m advai_notebooks.embedding run --config configs/embedding/smoke.yaml
python -m advai_notebooks.embedding evaluate --config configs/embedding/smoke.yaml \
  --checkpoint outputs/embedding/smoke/export --output outputs/embedding/re-evaluation.json
```

Use a fresh `output_dir` for every CLI run. Notebook executions create unique
folders. Copy `configs/embedding/bge.yaml` for a real model, point its three data
paths at your [corpus/queries/qrels](../../data/sample/retrieval/README.md), and
adjust the model, prefixes, maximum length and memory settings.

## Independent input and adjustment points

The trainer reads `data.training_samples`, a JSON artifact or JSONL file of
`{query_id, positive_id, negative_ids}` rows. IDs reference this experiment's
corpus and train queries. Every positive must have a positive qrel; negatives
must be judged zero unless explicitly opted into unjudged negatives. Prepared
embedding rows use an equal explicit-negative count. A sample artifact records
its kind, schema version and dataset fingerprint. Its contents are revalidated
before training; artifact metadata cannot override data policies.

Use `advai-data prepare --config configs/data_preparation/smoke.yaml` to prepare
samples separately, or supply your own valid file. Neither mining nor ranker
training is performed by `embedding run`. The smoke config reads a checked-in
prepared fixture. Domain presets read `data/local/retrieval/prepared/`.

Adjust embedding-specific settings and behavior in `config.py`, `data.py`,
`sampling.py`, `losses.py`, `train.py`, `metrics.py` and `evaluate.py`. The model adapter owns
preprocessing, loading/saving and inference; the pipeline assembles the experiment.
`backbones.py` owns the encoder component view; `heads.py` owns pooling, preserved
Dense/output modules and normalization. Inference and contrastive training use the
same underlying module objects, with original parameter names and native exports.

## Training and data policy

This version supports ordinary text SentenceTransformer encoders with one
pooling module and one normalized vector per text. `pooling: model` preserves
the checkpoint's pooling; `mean` and `cls` override it explicitly. Query and
document prefixes are applied once in training and inference. Exports preserve
prefixes as native SentenceTransformer query/document prompts and include a
normalization module. The BGE configuration uses its documented Chinese query
instruction and the checkpoint's pooling.

When the optional preparation stage mines hard negatives, they come from training judgments only. By default a negative must
have explicit relevance zero; unknown documents are not silently labeled zero.
`allow_unjudged_negatives: true` opts into that assumption within the training
document pool. Known positives and identical positive document text are excluded.
Queries are split by source group, and duplicate query text cannot cross splits.
Optional `enforce_document_groups` also isolates positive document source groups
and removes held-out positive groups from the mining corpus. Shared deployment
corpora may use new-query evaluation without document isolation; choose the
policy that matches the intended application.

The preparation tool creates one row per positive judgment; the trainer consumes the supplied rows. MNRL batches avoid shared query
groups, duplicate candidate text, and known positives appearing as another
query's negatives. This can make actual batch sizes smaller than configured.
Singleton batches require explicit hard negatives; a batch with no contrastive
signal fails early. No sampler can exclude undiscovered positive relationships;
inspect mined examples and judgment coverage for your domain.

`loss: mnrl` uses MultipleNegativesRankingLoss. `cached_mnrl` uses its official
gradient-cache implementation and `cached_mini_batch_size`. Gradient accumulation
accumulates several contrastive objectives; it does not expand each objective's
negative pool. Full encoder fine-tuning, gradient checkpointing, gradient
clipping and CUDA AMP are supported. `auto` selects CUDA, then MPS, then CPU.
CPU/MPS use fp32; CUDA supports fp16 and compatible bf16. Multi-GPU, LoRA and
multi-vector/decoder embeddings are outside this first version.

Inputs that exceed the configured sequence budget fail by default. Set
`reject_truncation: false` to accept the model's truncation for training and
evaluation; CLI inference remains strict. For Python inference, call
`check_lengths` before `encode` when truncation should fail. Set a realistic sequence limit
within the backbone's position capacity.

## Evaluation and export

Retrieval uses exact dot product over normalized vectors with chunked score
matrices and stable document-ID tie breaking. It searches the complete corpus,
without adding positive answers to candidate sets. Memory for score matrices is
bounded by `corpus_chunk_size × query_chunk_size`; query vectors, data and top-k
results remain in memory. Larger production indexes can be introduced behind
this interface later.

Reports include Recall@k, MRR@k and graded nDCG@k. Relevance greater than zero is
positive; nDCG uses `2**relevance - 1`. The denominator uses all known positives,
including those not recalled. Unjudged documents have zero gain for offline
metrics, which can understate quality with incomplete judgments. The validation
metric chooses `best/`, including epoch zero as a fallback if training never
improves it. The test split is only reported. A selected epoch-zero export is
explicitly recorded as `best_epoch: 0`; it does not demonstrate a training gain.

```text
config.json, environment.json, data_summary.json
training_samples.json, history.json
baseline/    # Pre-training native checkpoint
best/        # Best validation checkpoint; may be epoch zero
latest/      # Final completed-epoch native checkpoint
export/      # Selected model, tokenizer, prompts, pooling and metadata
baseline_test.json, test_rankings.json, report.json
```

Native directories serve inference; `latest/training_state.pt` also stores
completed-epoch optimizer/scaler/RNG state. Resume with the original config and
`--resume <output_dir>/latest`; source/config/data changes require a new run.
For a warm start, use `model.id` pointing to
`latest/` or `export/`, preserving the exported prefixes and sequence settings.
This starts a new optimizer and experiment history.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("outputs/embedding/smoke/export")
queries = model.encode_query(["reset password"])
documents = model.encode_document(["account login password reset"])
scores = queries @ documents.T
```

For strict wrapper inference:

```python
from advai_notebooks.embedding.model import EmbeddingModel
model = EmbeddingModel.load("outputs/embedding/smoke/export", device="cpu")
model.check_lengths(["reset password"], role="query")
vectors = model.encode(["reset password"], role="query")
```

The CLI `encode` command accepts a JSON list of strings, `--role query|document`,
`--checkpoint`, `--input` and `--output` (NumPy `.npy`).

## Validation and Colab

CPU tests run both real MNRL implementations with random tiny checkpoints,
verify weight updates and SDK export agreement, exercise leakage/negative rules,
and compare chunked retrieval with full scoring. Synthetic quality scores are
not a domain benchmark. GPU and large Hub models require target-hardware checks.

In Colab, place the repository at `/content/advai-notebooks`, run
`%pip install -e '/content/advai-notebooks[embedding,notebooks]'`, and set the
notebook's `ROOT` if necessary. Use a persistent Drive output path for long
experiments and restartable model warm starts. Keep real data out of Git.

API references: [training overview](https://sbert.net/docs/sentence_transformer/training_overview.html),
[MNRL and Cached MNRL](https://sbert.net/docs/package_reference/sentence_transformer/losses.html),
[BGE model card](https://huggingface.co/BAAI/bge-small-zh-v1.5).

## Metric report contract

The task-owned `metrics.py` returns the existing `metrics`, `per_query` and
`n_queries` fields, plus `task`, `metric_schema_version: 1` and `metric_records`.
Each record contains `name`, `value`, `n`, `group` and `direction`. Ranking metrics
maximize their value; overall records average queries, and per-query records have
`n: 1`. Full known qrels define recall denominators and nDCG ideals, including
positives absent from the results. Changing a task policy belongs in its local
metrics implementation. Formulas, record validation and selection rules are
owned by this task, with no shared metrics dependency.
