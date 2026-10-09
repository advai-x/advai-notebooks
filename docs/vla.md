# VLA — Vision-Language-Action Models

Status: guide outline and directory scaffold. Training, inference, CLI commands,
dependency extras and runnable notebooks are not implemented yet.

This track will cover instruction-conditioned action prediction from visual observations
and, when required, robot state. The first backend must specify its observation layout,
action representation, control frequency and prediction horizon.

## Planned workflow

1. Select a checkpoint and define supported cameras, robot state and action space.
2. Validate episodes, instructions, timestamps and observation/action alignment.
3. Fit state/action normalization on training episodes and preserve its statistics.
4. Train on demonstration actions using the backend's discrete-token or continuous-action
   objective, with an explicit freeze/full/adapter policy.
5. Evaluate on held-out episodes and report action errors in the defined coordinate system.
6. Where a simulator is available, evaluate rollout task success separately from offline loss.
7. Export and reload weights, processors, normalization statistics and action metadata.

## Planned data contract

Episode records will identify the episode, source group, split and language instruction.
Steps will include ordered camera observations, timestamps, optional robot state and
action labels or action chunks.

The backend must define action units, coordinate frames, joint ordering, gripper encoding,
chunk horizon and terminal markers. Entire episodes must stay in one split; additional
task, scene and robot grouping will depend on the intended evaluation.
The schema will be finalized and tested with the first executable backend.

## Implementation locations

| Area | Directory |
| --- | --- |
| Notebook | [notebooks/vla](../notebooks/vla/README.md) |
| Configuration | [configs/vla](../configs/vla/README.md) |
| Python package | [src/advai_notebooks/vla](../src/advai_notebooks/vla/) |
| Sample data | [data/sample/vla](../data/sample/vla/README.md) |
| Tests | [tests/vla](../tests/vla/README.md) |
| Outputs | [outputs/vla](../outputs/vla/README.md) |

The first backend should include tiny synthetic episodes, alignment/normalization checks,
and export/reload validation. Offline action error and simulator success are distinct
evaluation results; any robot or simulator integration will be documented explicitly.
