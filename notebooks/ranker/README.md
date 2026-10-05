# Independent ranker fine-tuning

[01_finetune.ipynb](01_finetune.ipynb) now runs ranker only. No embedding model
is imported or loaded by ranker training, evaluation or inference.

```bash
python -m pip install -e '.[ranker,notebooks,dev]'
python -m advai_notebooks.ranker validate --config configs/ranker/smoke.yaml
python -m advai_notebooks.ranker run --config configs/ranker/smoke.yaml
python -m advai_notebooks.ranker evaluate --config configs/ranker/smoke.yaml \
  --checkpoint outputs/ranker/smoke/export --output outputs/ranker/re-evaluation.json
```

Each CLI run needs a new output folder. Notebook runs use unique folders.
`data.training_samples` supplies ranker pairs; `data.candidates` supplies fixed
validation/test candidates. The smoke preset reads checked-in synthetic files.
`bge.yaml` loads a real one-logit classifier and user-prepared data, with no
retriever settings in the ranker config.

## Inputs and validation

A training row is `{query_id, doc_id, label}`, where label is numeric zero or one.
IDs reference this experiment's corpus and train queries. Positive rows need a
positive qrel; negative rows need relevance zero unless unknown negatives are
explicitly opted into. Training rejects held-out queries/source documents,
duplicate pairs, conflicting known-positive labels and missing train queries.
Both classes must be present. Use JSONL rows or the versioned JSON artifact
written by the preparation stage.

Candidates are a JSON object with `schema_version: 1`, `data_fingerprint` and
`splits`, containing validation/test query IDs mapped to document/score rows.
See [the data contract](../../data/sample/retrieval/README.md). Candidates must
cover the requested split exactly, contain no duplicate/unknown documents and
have finite scores. Positives missing from the candidate set stay missing.

The ranker can read candidates from an existing embedding, keyword retriever or
manual system through this file contract. No retriever is executed implicitly.
For re-evaluation, the configured file is used by default; `--candidates` can
explicitly select another compatible artifact, including a saved run's file.

## Fine-grained adjustment points

- `config.py`: ranker-only configuration and supported settings.
- `data.py`: pair labels and candidate contracts.
- `sampling.py`: pair shuffling/batching.
- `losses.py`: relevance objective; currently pointwise BCEWithLogitsLoss.
- `train.py`: sampler, loss and validation callback assembly.
- `evaluate.py`: candidate scoring and evaluation orchestration.
- `metrics.py`: independent ranking formulas, record validation and selection rules.
- `backbones.py`: pair encoder view and the native classification forward path.
- `heads.py`: native classifier-side module views and the raw relevance-logit rule.
- `model.py`: component assembly, model loading, text preprocessing, inference and native exports.
- `pipeline.py`: experiment execution and artifact writing.
- `cli.py`: independent commands.

The neutral optimizer loop handles AdamW, weighted gradient accumulation,
clipping, checkpoint saving and validation selection. Embedding objectives and
samplers have their own modules. Current BCE collapses positive grades into
label one; nDCG evaluation keeps the grades. Multi-logit classifiers and
generative rerankers remain outside this first version.

Component views reuse the original HF modules without registering parameters twice.
`model.backbone` exposes the encoder and `model.head.modules` exposes classifier-side
modules. Scoring retains the native full classifier forward, including architecture-specific
pooling/dropout order; the adapters do not replace it with a generic classifier.
Changes to scoring layers belong in this task's `heads.py`, with matching configuration
and export/reload support when the network architecture changes.

## Outputs and inference

Outputs include `training_pairs.json`, `candidates.json`, `history.json`,
`environment.json`, `baseline/`, `best/`, `latest/`, `export/`, and `report.json`.
Validation selects the export, including epoch zero as a fallback. The test
split is reported only. `latest/training_state.pt` stores epoch-boundary
optimizer/scaler/RNG state; resume with unchanged config/data/source using
`--resume <output_dir>/latest`.

```python
from advai_notebooks.ranker.model import RankerModel
model = RankerModel.load("outputs/ranker/smoke/export", device="cpu")
pairs = [("reset password", "account login password reset")]
model.check_lengths(pairs)
scores = model.score(pairs)
```

Sort raw logits descending; they are not calibrated probabilities. Prefixes are
handled by the wrapper. Native `sentence_transformers.CrossEncoder` can also
load the export; apply metadata prefixes once and request
`activation_fn=torch.nn.Identity()` for raw logits.

`advai-ranker rerank` accepts a JSON list of `[query, document]` pairs and writes
scores, with `--checkpoint`, `--input`, `--output`, and optional `--device`.
Hardware, truncation and group-splitting rules follow the embedding guide.
Existing exports retain the same inference API.

## Optional preparation and combined evaluation

`advai-data prepare --config configs/data_preparation/smoke.yaml` prepares samples
without a model by default. Point the ranker data paths at the returned files.
An optional preparation encoder may mine harder negatives; it runs before
training as a separate step.

Combined embedding/ranker evaluation lives in
[the workflow notebook](../workflows/01_retrieval_comparison.ipynb) and
`advai-retrieval compare --config configs/workflows/smoke.yaml --embedding-run ...
--ranker-run ... --output ...`. Each retriever's candidates are shared between
both rankers. The combined workflow is not part of either trainer.

CPU tests verify weight updates, candidate membership, SDK agreement and
training/inference with embedding imports explicitly blocked. Synthetic smoke
scores are not domain quality; large-model and GPU validation remain.

## Metric report contract

The task-owned `metrics.py` returns the existing `metrics`, `per_query` and
`n_queries` fields, plus `task`, `metric_schema_version: 1` and `metric_records`.
Each record contains `name`, `value`, `n`, `group` and `direction`. Ranking metrics
maximize their value; overall records average queries, and per-query records have
`n: 1`. Full known qrels define recall denominators and nDCG ideals, including
positives absent from the results. Changing a task policy belongs in its local
metrics implementation. Formulas, record validation and selection rules are
owned by this task, with no shared metrics dependency.
