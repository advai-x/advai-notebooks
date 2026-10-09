# VLA — Vision-Language-Action Model

English | [简体中文](vla.zh-CN.md)

This track implements a CLIP + state + continuous action-chunk behavior-cloning
baseline. Separate CLIP image/text encoders process the observation and instruction;
their features are concatenated with normalized state to predict the next H actions.
You can freeze pretrained encoders and train only the head, or update encoders and
head together. **This is not an OpenVLA/SmolVLA checkpoint adapter or a validated
robot controller.**

## Quick start

```bash
python -m pip install -e '.[vla,notebooks,dev]'
python -m advai_notebooks.vla validate --config configs/vla/smoke.yaml
python -m advai_notebooks.vla run --config configs/vla/smoke.yaml
# Update the tiny encoders as well as the action head
python -m advai_notebooks.vla run --config configs/vla/full.yaml
```

The [notebook](../notebooks/vla/01_finetune.ipynb) includes saved CPU outputs.
Tiny CLIP and the action head are randomly initialized. The real encoder template
in [hf.yaml](../configs/vla/hf.yaml) pins an `openai/clip-vit-base-patch32` revision;
its action head still starts from random parameters.

## Episode data contract

Use one episode per JSONL line. Episodes/groups cannot cross train, validation or test:

```json
{"id":"episode-1","group_id":"session-1","split":"train","instruction":"move right","steps":[{"timestamp":0.0,"image":"images/frame-0.png","state":[0.0,0.0],"action":[0.1,0.0]},{"timestamp":0.1,"image":"images/frame-1.png","state":[0.1,0.0],"action":[0.1,0.1]}]}
```

- All three splits must be nonempty; IDs must be unique. Group related episodes from the same session/robot trajectory yourself.
- Instructions must be nonempty. Every step needs finite timestamps, state/actions and a decodable local image.
- Images are relative to the JSONL directory and cannot escape it. Duplicate decoded images cannot cross splits.
- Timestamps must increase strictly at intervals matching `1 / data.control_frequency_hz`. Irregular sampling is unsupported.
- Explicitly match `state_dim`/`action_dim`, dimension names/units, action coordinate frame and control frequency.
- Each action follows that step's image/state observation. Sample t predicts actions t through t+H−1.
- Action chunks never cross episode boundaries. A mask excludes tail padding from training and evaluation.
- Instructions exceeding `max_len` fail without silent truncation; the native CLIP model defines the context ceiling.

State/action means and standard deviations are fitted only on each original training
step, not counted repeatedly through overlapping chunks. Exports retain these
statistics and physical semantics. Evaluation and inference reuse them without refitting.

## Loss, metrics and selection

Training uses masked MSE in normalized action space. Evaluation reports action
MSE/MAE after denormalization, weighted across valid horizon/action scalars.
Lower `validation.action_mse` is better; selection includes the epoch-zero baseline.
Gradient accumulation is weighted by valid action scalars. Test never selects models.

```bash
python -m advai_notebooks.vla run --config configs/vla/smoke.yaml \
  --resume outputs/vla/smoke/latest
python -m advai_notebooks.vla evaluate --config configs/vla/smoke.yaml \
  --checkpoint outputs/vla/smoke/export --output outputs/vla/test.json
```

Training is single-process, single-device fp32. Epoch checkpoints retain optimizer/RNG
state and strictly verify configuration, data/media, source and dependency versions.

## Action inference, export and publication

```python
from advai_notebooks.vla.model import VLAModel

model = VLAModel.load("outputs/vla/smoke/export")
prediction = model.predict("data/sample/vla/images/cyan.ppm", "move up", [0.0, 0.1])
# prediction["actions"]: H × action_dim, in original physical units
# prediction["action_spec"]: names, units, frame, frequency and horizon
```

CLI `predict --checkpoint ... --input requests.json --output actions.json` accepts:

```json
[{"image":"data/sample/vla/images/cyan.ppm","instruction":"move up","state":[0.0,0.1]}]
```

Inference image paths are relative to the working directory. Predictions are not
automatically clipped or executed on a robot.

`export/` saves the selected encoder, processor/tokenizer, action-head safetensors,
training normalization and action semantics. `latest_export/` is the final epoch.
Reloading does not depend on the original CLIP download location.

```bash
python -m advai_notebooks.vla export --checkpoint outputs/vla/smoke/export \
  --output outputs/vla/copied
python -m advai_notebooks.vla publish --export-dir outputs/vla/copied \
  --repo-id your-org/your-vla --dry-run
```

Publication filters and reloads inference files, excluding episodes, images and
training state. The dry-run preview does not access the Hub.

## Validation scope and limitations

CPU tests cover frozen/full encoders, initialization from a local native CLIP base,
tail masks, train-only normalization, time/media/group contracts, action inference
reload, recovery and publication previews. Robot fine-tuning with actual CLIP Hub
weights, CUDA/MPS and closed-loop robot success rates have not been verified.
There is no discrete action tokenizer, diffusion action head, LoRA, LeRobot/RLDS
adapter or robot connection.

Offline MSE does not establish safe control or task success. Real deployment needs
action bounds, hardware safety controls, coordinate transforms and closed-loop
evaluation. Check robot-data authorization and base-model licenses separately.

Official encoder reference: [CLIP](https://huggingface.co/docs/transformers/model_doc/clip).
