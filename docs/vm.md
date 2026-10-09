# VM — World Models

Status: guide outline and directory scaffold. Training, inference, CLI commands,
dependency extras and runnable notebooks are not implemented yet.

`vm/` means **World Model** in this project. This track will cover learning dynamics
from sequences of observations and, where available, actions, then predicting future
states or observations over a defined horizon.

## Planned workflow

1. Select a backend and define whether it predicts pixels, latent representations or states.
2. Validate ordered episodes, context windows, prediction targets and optional actions.
3. Fit preprocessing and normalization on the training split and construct temporal batches.
4. Train with the backend's transition, reconstruction or generative objective.
5. Evaluate both one-step predictions and multi-step rollouts on held-out episodes.
6. Report prediction horizon, rollout conditioning and temporal error alongside sample outputs.
7. Export and reload the dynamics model, observation processor, decoder when needed,
   and rollout configuration.

## Planned data contract

Episode records will identify the episode, source group, split and ordered observations.
Each step will carry timestamps and, for action-conditioned training, the action that
links the current observation to the next one. Rewards and terminal markers are optional
and must have an explicit meaning when used.

Context/target windows must stay within episode boundaries. Overlapping windows from
the same episode must not cross data splits. Evaluation must describe whether rollouts
use predicted observations or ground-truth context and which actions they receive.
The schema will be finalized and tested with the first executable backend.

## Implementation locations

| Area | Directory |
| --- | --- |
| Notebook | [notebooks/vm](../notebooks/vm/README.md) |
| Configuration | [configs/vm](../configs/vm/README.md) |
| Python package | [src/advai_notebooks/vm](../src/advai_notebooks/vm/) |
| Sample data | [data/sample/vm](../data/sample/vm/README.md) |
| Tests | [tests/vm](../tests/vm/README.md) |
| Outputs | [outputs/vm](../outputs/vm/README.md) |

The first backend should include a tiny synthetic dynamics fixture, temporal-boundary
tests, rollout evaluation and export/reload checks. VLA predicts actions, whereas VM
predicts what follows from observations and optional actions. Their episode artifacts
may be shared when the data contracts are compatible.
