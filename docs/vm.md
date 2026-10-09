# VM — World Model

English | [简体中文](vm.zh-CN.md)

In this repository, VM means World Model. This track implements an action-conditioned
residual MLP in numerical state space: current observation + action → next observation,
with recursive multi-step prediction. Train tiny dynamics from random initialization
or fine-tune an existing native VM export. This is a small dynamics baseline,
**not a video world model, Dreamer implementation or robot planning system**.

## Quick start

```bash
python -m pip install -e '.[vm,notebooks,dev]'
python -m advai_notebooks.vm validate --config configs/vm/smoke.yaml
python -m advai_notebooks.vm run --config configs/vm/smoke.yaml
# Fine-tune the preceding export in a new output directory
python -m advai_notebooks.vm run --config configs/vm/finetune.yaml
```

The [notebook](../notebooks/vm/01_finetune.ipynb) executes initial training, native
weight fine-tuning and rollout. Synthetic trajectories use `next_state = state + action`;
metrics are engineering regression checks, not evidence of real-world modeling quality.

## Episodes and temporal alignment

Each JSONL line contains one episode: N observations/timestamps and N−1 actions.
Action t connects `observations[t]` to `observations[t+1]`.

```json
{"id":"trajectory-1","group_id":"session-1","split":"train","timestamps":[0.0,0.1,0.2],"observations":[[0.0,0.0],[0.1,0.0],[0.1,0.2]],"actions":[[0.1,0.0],[0.0,0.2]]}
```

- Train/validation/test must all be nonempty, with unique IDs. Groups and identical trajectories cannot cross splits.
- Observation/action dimensions must match the model; values must be finite. Each episode needs at least two observations.
- Timestamps must increase strictly at `1 / data.control_frequency_hz` intervals. Only fixed time steps are supported.
- State/action names, units and control frequency are part of the exported model contract. Arbitrary image/video inputs are unsupported.
- Horizon windows cannot cross episode boundaries; masks exclude tail padding.
- Normalization counts each original training observation/action once, not repeatedly through overlapping windows.

## Model and training objective

The MLP concatenates normalized observations/actions and predicts an observation
delta. Set `predict_delta: false` to predict the normalized next state directly.
Training starts from the actual current observation, then recursively feeds back its
own predictions rather than ground-truth future states. Loss is masked MSE over
valid horizon/state scalars; gradient accumulation uses the same scalar weighting.

Evaluation reports `state_mse` (one step) and `rollout_mse` (multiple steps) in
original units. Predictions are conditioned on a supplied ground-truth action
sequence; **this is not closed-loop action planning**. Selection minimizes validation
rollout MSE, including the baseline. Test data does not select models.

## Fine-tuning versus checkpoint recovery

`model.id` can point to a native local VM export or Hub repository; a Hub base requires
a pinned revision. Observation/action dimensions, `hidden_size` and `predict_delta`
must match the base. Base normalization is preserved so learned weights retain their
physical meaning. Data names, units and frequency must also match. Start a new
experiment when changing training data or settings; do not use `--resume` for this.

```bash
python -m advai_notebooks.vm run --config configs/vm/smoke.yaml \
  --resume outputs/vm/smoke/latest
python -m advai_notebooks.vm evaluate --config configs/vm/smoke.yaml \
  --checkpoint outputs/vm/smoke/export --output outputs/vm/test.json
```

Training is single-process, single-device fp32. Checkpoints save optimizer/RNG state;
recovery requires unchanged configuration, data, source and relevant dependencies.

## Predict, rollout and inference packages

```python
from advai_notebooks.vm.model import VMModel

model = VMModel.load("outputs/vm/smoke/export")
next_state = model.predict([0.0, 0.0], [0.1, 0.0])
future = model.rollout([0.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
# future["observations"]: len(actions) × observation_dim
```

Action sequences must be nonempty and no longer than the model's horizon.
CLI `rollout --checkpoint ... --input requests.json --output future.json` accepts:

```json
[{"observation":[0.0,0.0],"actions":[[0.1,0.0],[0.0,0.1]]}]
```

`export/` contains the selected model; `latest_export/` contains the final epoch.
Inference packages include safetensors weights and metadata describing architecture,
normalization and state/action semantics. Training trajectories are not required.

```bash
python -m advai_notebooks.vm export --checkpoint outputs/vm/smoke/export \
  --output outputs/vm/copied
python -m advai_notebooks.vm publish --export-dir outputs/vm/copied \
  --repo-id your-org/your-world-model --dry-run
```

The preview actually reloads the package but does not access the Hub. Trajectories,
optimizer state and logs are excluded.

## Validation limits

CPU regression tests cover episode alignment, padding masks, train-only normalization,
recursive multi-step gradients, inference reload, fine-tuning existing VM weights,
strict recovery and publication previews. CUDA/MPS, real robot/game environments,
video/image prediction, latent stochastic dynamics, reward/done prediction and
planning have not been verified or implemented as applicable.
Real applications need checks of the Markov-state assumption, accumulated error at
different horizons and out-of-distribution trajectories. This baseline has no
recurrent hidden state.
