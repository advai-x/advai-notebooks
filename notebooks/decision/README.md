# Native decision fine-tuning

Open [01_finetune.ipynb](01_finetune.ipynb) for training, calibration, evaluation,
export and independent inference. The default smoke experiment uses a random tiny
BERT and our own option scorer on 30 synthetic customer records. It checks the
pipeline; its scores do not measure domain quality.

## Install and run

```bash
python -m pip install -e '.[decision,notebooks,dev]'
python -m advai_notebooks.decision validate --config configs/decision/smoke.yaml
python -m advai_notebooks.decision run --config configs/decision/smoke.yaml
python -m advai_notebooks.decision evaluate --config configs/decision/smoke.yaml --model outputs/decision/smoke/export --split test
python -m advai_notebooks.decision predict --model outputs/decision/smoke/export --task triage --device cpu --state 'charged twice refund today or I cancel'
```

For domain training, copy `configs/decision/hf.yaml` and point data/tasks at your
own files. `model.id` is a Hugging Face base-encoder ID or local encoder directory;
`revision` and `subfolder` are optional. The example uses multilingual BERT;
local tiny BERT/RoBERTa compatibility has been tested, while real Hub checkpoints
still need validation on your data/hardware. The decision head initializes randomly
when loading a base encoder. This does not provide trained zero-shot decisions.

## Model and adjustment points

```text
state + question + each option → HF encoder → mean/CLS pooling
                              → type embedding + shared MLP → option logits
                              → softmax / optional temperature → typed answer
```

The encoder runs per option, so compute grows with the option count. Question type
is explicit, and the head supports variable option counts. There is no auxiliary
action head or external decision SDK. `model.pooling`, `head_hidden_size`,
`dropout`, `freeze_encoder`, separate encoder/head learning rates and optimizer are
configurable. See `decision/contracts.py`, `backbones.py`, `heads.py`, `encoding.py`, `decoding.py`,
`optimizer.py`, `losses.py` and `calibrate.py` for independent changes. The supported
config components are `encoding: pair`, `head: independent`, `decoder: typed`.
See the [architecture guide](../../docs/architecture.md) for contracts and extension
requirements; cross-option attention is a future experiment.

Use a bidirectional text encoder exposing `hidden_size`, a fast tokenizer and a
pad token. BERT/RoBERTa pair conventions are applied by the tokenizer. Decoder and
seq2seq models are rejected. `head_max_len` caps each question/option segment;
`max_len` caps the complete state pair including special tokens. Long or
indistinguishable tokenized options are rejected. State truncation is rejected by
default; `data.reject_truncation: false` permits it and reports affected decisions.
Inference rejects truncation. More context does not by itself establish accuracy.

## Data and labels

Task JSON maps task IDs to questions. `choice` uses a label-to-description criteria
object, `score` an ordered list of levels and `noul` false/true. Records retain the
existing contract:

```json
{"id":"ticket-1","group_id":"source-1","split":"train","task":"triage","state":"charged twice, please refund","gold":{"department":{"choice":"billing"},"urgency":{"score":1},"churn_risk":{"noul":false}}}
```

Hard score labels are zero-based integer levels. Soft labels use
`{"probabilities":{"billing":0.8,"technical":0.2}}`; missing options receive
zero mass and finite positive totals are normalized. Unknown options, negative
or non-finite probabilities and zero total mass are rejected. Partial question
labels and JSON object/list states are supported.

Source groups and identical states cannot cross splits. Use train/validation/test
and, when calibration is enabled, a separate calibration split. Similarity/time
leakage still requires a domain-specific split policy.

## Training, selection and calibration

Default `loss: supervised` combines soft cross-entropy for all types with normalized
ranked probability score for ordered `score` levels, weighted by `ce_weight` and
`ordinal_weight`. `loss: ce` uses only CE as an ablation. `loss: rlcd` adds Gaussian-logit policy-gradient
training with native proper scores: negative Brier for choice/noul and negative
normalized ranked probability score for ordered levels. CE/ordinal/RL weights and noise
samples/sigma schedule are configurable. Setting `ordinal_weight: 0` restores the
earlier CE + RLCD objective. The reward differs from the old adapter. Compare
`smoke.yaml`, `ce_smoke.yaml` and `rlcd_smoke.yaml` with matching conditions; these
synthetic runs do not establish that RLCD improves domain quality.

Validation selects by NLL/Brier/ECE (minimize) or accuracy (maximize).
`evaluation.include_baseline: true` allows epoch zero to win if training does not
improve the selected metric. Test data never selects weights or temperatures.

`calibration.enabled: true` fits bounded temperatures on calibration data. Defaults
are `min_type_samples: 10`, `min_bucket_samples: 2000`; sparse types use temperature
1, sparse buckets fall back to their type. The fit includes identity as a candidate,
so calibration-set NLL cannot regress, but held-out test quality can. Tune minimum
counts for your data. Disabling calibration removes the extra split requirement,
exports identity temperatures and reports calibrated metrics as null.

One CPU/MPS/CUDA device is supported. Auto selects CUDA, then MPS, then CPU;
CPU/MPS require fp32; CUDA also supports fp16/bf16 where available. Gradient
accumulation, clipping and encoder gradient checkpointing are supported. Multi-GPU
and LoRA are not implemented.

## Export, inference and resume

```text
config.json, environment.json, data_summary.json
baseline_metrics.json, baseline_predictions.json, history.json
best/, latest/                # Native weights, configuration and tokenizer
latest/training_state.pt      # Atomic epoch model/best/optimizer/AMP/RNG state
export/                       # Native model + tasks and calibration metadata
predictions.json, report.json
```

Exports contain `decision_config.json`, `model.safetensors`, `encoder/`,
`tokenizer/`, `source.json`, `tasks.json` and `calibration.json`:

```python
from pathlib import Path
from advai_notebooks.decision.model import DecisionModel

path = Path("outputs/decision/smoke/export")
model = DecisionModel.load(path, device="cpu")
result = model.system_one("charged twice", task="triage")
```

For direct inference without a task file or labels:

```python
from advai_notebooks.decision.data import DecisionOption, DecisionRequest

request = DecisionRequest("charged twice", "Which department?", (
    DecisionOption("billing", "Payment or refund"),
    DecisionOption("technical", "Service outage or bug"),
))
answer = model.system_one([request])[0]
```

Choice returns the highest-probability label; score returns expected zero-based
level plus the argmax level; noul returns true probability in [0, 1]. All three
return `probabilities` and `confidence=max(p)`; `answer_confidence` remains an
equivalent compatibility field. For noul, confidence can belong to false and
is distinct from the true probability. `min_confidence` adds an abstained flag;
application code must decide whether to act on a flagged answer.

Resume the original experiment with:

```bash
python -m advai_notebooks.decision run --config configs/decision/smoke.yaml --resume outputs/decision/smoke/latest
```

Keep the original configuration/planned epoch count, data, dependencies, device
and relevant source unchanged. Resume restores completed-epoch weights, best
weights, optimizer/AMP and Python/NumPy/Torch/device RNG; a partial epoch is replayed.
Native inference weight files damaged after a committed epoch can be repaired from
training state. Old external SDK weights are incompatible with the new architecture;
retrain them. Native v1 option-scorer inference exports migrate to v2 without
changing probability calculations; earlier training states require unchanged
implementation and cannot resume through this refactor. Task data and typed
responses remain compatible.

## Validation and Colab

```bash
python -m pytest tests/decision -q
```

Tests cover CE/ordinal/RLCD gradients, label-free requests, replaceable components,
native v1 migration, token budgets, option-order/padding invariance,
calibration and exact resume with dropout, and block external decision/retrieval
packages during real training/inference/CLI execution. Large-model/CUDA/MPS behavior
must be checked on target hardware. Colab uses this repository at
`/content/advai-notebooks`; enable the notebook dependency cell and use persistent
storage for long experiments. Keep domain data and generated weights outside Git.

## Aligned public boundaries

Decision now exposes `DecisionModel.load/save/from_config`, `prepare/features/check_lengths`
and `system_one` in `model.py`, alongside embedding/ranker's model adapters.
State/task inference automatically uses the loaded export's `tasks.json`; old
predict methods remain compatibility aliases.
Request types, task validation and labeled data assembly enter through `data.py`.
`evaluate.py` returns `(metrics, predictions)` and owns optional calibration fitting
and selected-model evaluation. The pipeline assembles these stages and writes artifacts.

Initialization, loading and saving are implemented directly in `model.py`;
there is no separate provider directory or Bundle layer. All three tasks now use task-local `backbones.py` and `heads.py`; Decision retains
its own option encoder and scoring head. Schema, encoder and decoder remain
replaceable internal components. Existing predict/function APIs
delegate to the model facade. Native v1/v2 weights
and probability semantics remain compatible; pre-refactor training states still obey
strict source fingerprint checks.

Decision metric reports retain `overall`, `by_question` and `by_type` and add
`task`, `metric_schema_version: 1`, and `metric_records`. The records follow the same
name/value/sample-count/group/direction format as embedding/ranker; formulas,
record validation and selection rules are implemented independently in this task. Accuracy
maximizes; NLL/Brier/ECE/ordinal error minimize. For `normalized_score_mae`,
`n` counts score decisions only, even inside a mixed overall group. Selection
reads the unique overall record; the configured selection metric options remain
unchanged.
