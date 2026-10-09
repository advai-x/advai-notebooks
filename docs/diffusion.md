# Diffusion image models

English | [简体中文](diffusion.zh-CN.md)

`diffusion/` is a peer of Embedding, Ranker, Decision and LLM, with independent
training, metrics, inference and publication. The first backend supports Stable
Diffusion v1/v2-style latent text-to-image models: one CLIP text encoder, VAE and
UNet. Diffusers 0.41.0 provides native components and serialization; this project
owns training, data contracts, evaluation and selection. Its text encoder is an
internal conditioning component, not the project's Embedding model.

## Install and run offline

```bash
python -m pip install -e '.[diffusion,notebooks,dev,publish]'
python -m advai_notebooks.diffusion validate --config configs/diffusion/smoke.yaml
python -m advai_notebooks.diffusion run --config configs/diffusion/smoke.yaml
python -m advai_notebooks.diffusion run --config configs/diffusion/lora.yaml
```

These configs run full UNet and LoRA training respectively. Tiny uses randomly
initialized UNet, CLIP text encoder and VAE, a local WordLevel tokenizer and a noise
scheduler, without downloading a base. Eight synthetic 16×16 images check mechanics
only. The VAE/text encoder are not pretrained either; outputs do not demonstrate
image quality or text alignment. `run` rejects nonempty output directories. Choose
a new `output_dir` for another experiment, or use `--resume` for an interrupted run.

## Responsibilities and inputs

| Responsibility | Entry point |
| --- | --- |
| Configuration | `config.py` |
| Network components | `backbones.py`: native UNet/VAE/text encoder; `heads.py`: UNet output-layer view |
| Loading, saving, composition, inference | `model.py`: `DiffusionModel.load/save/generate` |
| Images and captions | `data.py` |
| Batches, noise, timesteps | `sampling.py` |
| Forward noise and inference schedules | `schedulers.py`, distinct from a learning-rate scheduler |
| Objective and optimizer | `losses.py`, `optimizer.py` |
| Training, metrics, evaluation | `train.py`, `metrics.py`, `evaluate.py` |
| Lifecycle, recovery, CLI, publication | `pipeline.py`, `runtime.py`, `cli.py`, `publish.py` |

Each JSONL line:

```json
{"id":"image-1","group_id":"asset-1","split":"train","image":"images/one.png","caption":"a red square"}
```

`image` is relative to the JSONL directory; absolute/escaping paths are rejected.
Fields are strictly validated and all three splits must be nonempty. Groups and
identical decoded RGB images cannot cross splits. Fingerprints cover original image
SHA256, decoded pixels and records. Images receive EXIF orientation correction,
RGB conversion, center cropping/resizing and normalization to [-1,1]. There is no
random image augmentation. Captions/inference prompts exceeding the text-encoder
budget fail without silent truncation. Identical captions do not imply duplicate images.

For real data, adapt `configs/diffusion/hf.yaml`: base ID/revision, resolution and
JSONL path. Remote bases resolve to immutable commit SHAs. Local bases can be native
complete pipelines or project full/merged exports. SDXL, multiple text encoders,
DiT/Flux, video, img2img and inpainting are unsupported; changing an ID does not add them.

## Training and evaluation

Full mode updates the entire UNet, not the entire pipeline. LoRA updates only UNet
attention adapters at `to_q/to_k/to_v/to_out.0`, with configurable rank/alpha.
Both freeze VAE and text encoder. Training samples VAE posterior latents, Gaussian
noise and uniform timesteps, adds noise, then computes mean MSE per image.
The base scheduler's `prediction_type` defines the target; `epsilon` or
`v_prediction` can be set explicitly. Learning rate is constant. AdamW/Adam,
accumulation, clipping and UNet gradient checkpointing are supported. Partial
accumulation windows normalize by their actual sample count.

Evaluation uses VAE posterior mode. Evaluation seed, sample ID and draw index fix
noise/timesteps. Per-image evaluation is invariant to training RNG and batch size;
`noise_draws` is the number of repeated evaluations per image. Task-owned
`metrics.py` implements `denoising_mse` and task/schema/name/value/n/group/direction
records. `n` counts images, not draws; selection minimizes MSE.

Lifecycle: validate → baseline validation/test → train → validation selection →
test → native export. Epoch zero participates; only strict improvements replace
best. Test never optimizes/selects. Fixed-seed test-caption generations are saved
under `samples/selected-test/` for qualitative checks. Lower denoising loss does not
prove better generation; FID/CLIP score are not implemented. There is no fitted
postprocessing/calibration stage.

## Artifacts and interrupted-run recovery

```text
output_dir/
  baseline/                           # Fixed initial model
  checkpoints/checkpoint-N/
    model/                            # Inference assets for that epoch
    state.pt                          # Optimizer + Python/NumPy/Torch/device RNG
    resume.json                       # Signatures, epoch, selection history
  latest.json                         # Atomic pointer to latest complete checkpoint
  export/                             # Validation-selected model; possibly epoch zero
  latest_export/                      # Last completed training epoch
  samples/selected-test/               # Fixed-seed images and images.json
  report.json config.json data_summary.json environment.json
```

```bash
python -m advai_notebooks.diffusion run \
  --config configs/diffusion/smoke.yaml --resume outputs/diffusion/smoke/latest
```

Resume from a complete checkpoint or `latest` alias in the same experiment, not an
inference export. Checks cover config (including planned epochs), image content,
actual base parameters, component configs, tokenizer, relevant source and dependencies.
Epoch checkpoints stage in `.incomplete` before atomic commit; pointers update last.
Incomplete epochs replay from the previous boundary. Inspect/clean a leftover
`.incomplete` directory before replay. Resuming a completed run can regenerate
exports without overwriting checkpoints. Only one process is supported;
distributed launches are explicitly rejected.

## Load, generate and export

```python
from advai_notebooks.diffusion.model import DiffusionModel

model = DiffusionModel.load("outputs/diffusion/smoke/export", device="cpu")
images = model.generate(["a red square pattern"], seed=123, num_inference_steps=4)
images[0].save("sample.png")
```

`generate` returns ordered PIL images for a prompt string or list. Override seed,
steps and guidance scale as needed. Each image uses an independent CPU generator,
not the global training RNG. Reproducibility requires the same environment and
batch order; pixel equality across hardware is not guaranteed. DDIM/DDPM inference
schedulers are configurable; training always uses the saved DDPM noise process.

```bash
python -m advai_notebooks.diffusion generate \
  --checkpoint outputs/diffusion/smoke/export \
  --input configs/diffusion/prompts.json --output outputs/diffusion/generated --seed 123
python -m advai_notebooks.diffusion evaluate \
  --config configs/diffusion/smoke.yaml --checkpoint outputs/diffusion/smoke/export \
  --split test --output outputs/diffusion/reloaded-test.json
python -m advai_notebooks.diffusion export \
  --checkpoint outputs/diffusion/lora/export --merged --output outputs/diffusion/merged
```

`generate --output` is a directory containing PNGs and `images.json`;
`evaluate --output` is a JSON file. Full exports contain native Diffusers
`model_index.json` and component weights/configs/tokenizer. LoRA exports contain
native `pytorch_lora_weights.safetensors` (including rank/alpha metadata) and a
frozen `base_model/`. Every LoRA package currently bundles the entire base for
offline reload, so it is larger than adapter-only packages. Merged export fuses and
unloads LoRA on a copy, leaving the training model unchanged.

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("outputs/diffusion/smoke/export")
# Native LoRA: load export/base_model, then pipe.load_lora_weights(export).
```

Native LoRA inference must also restore the trained scheduler contract. Read
`scheduler.inference` in `advai_metadata.json`, load the corresponding `DDIMScheduler`
or `DDPMScheduler` from exported `noise_scheduler/`, and assign `pipe.scheduler`.
The bundled base retains its original config; if `prediction_type` changed, reusing
that base scheduler would apply the wrong target. Project `load/generate` restores
this automatically.

## Explicit publication

```bash
python -m advai_notebooks.diffusion publish \
  --export-dir outputs/diffusion/smoke/export \
  --repo-id your-org/your-diffusion --dry-run
```

Only removing `--dry-run` uploads; new repositories default to private.
`--model-card` accepts a custom English card. The filtered package is actually
loaded before publication. It contains native inference components, LoRA/base,
noise config and a hash manifest, excluding source images, generated samples,
training state and logs. The implementation did not create a Diffusion Hub model
or Dataset. Publish/verify training datasets independently before linking them
from a real model card.

## Validation limits

CPU fp32 checks cover full/LoRA training, updates/freezing, native pixel-identical
reload, alpha preservation, merged export, deterministic evaluation, accumulation,
gradient checkpointing, `v_prediction`/DDPM scheduler reload, exact epoch recovery,
CLI, offline packages and import isolation. CUDA/MPS are configurable but not
verified on target hardware. AMP, multi-GPU, EMA, quantization and other model
families are unsupported. Real pretrained models/data require separate quality
and resource evaluation on appropriate hardware.

Official references: [text-to-image training](https://huggingface.co/docs/diffusers/training/text2image),
[LoRA training](https://huggingface.co/docs/diffusers/training/lora),
[native pipeline loading](https://huggingface.co/docs/diffusers/using-diffusers/loading).
