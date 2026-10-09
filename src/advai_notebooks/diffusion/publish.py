"""Diffusion-native component/LoRA file contract and an English publication card."""

import json
from pathlib import Path

from advai_notebooks.common.hub import card, collect_files, publish_files

from .model import DiffusionModel


def publish(export_dir, repo_id, *, license=None, model_card=None, **options):
    root = Path(export_dir).resolve()
    metadata = json.loads((root / "advai_metadata.json").read_text())
    if metadata.get("kind") != "diffusion":
        raise ValueError("Expected a diffusion inference export")
    # Only known component folders and native inference filenames are publishable.
    folders = ["", "noise_scheduler"]
    components = [
        "unet",
        "vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "safety_checker",
        "feature_extractor",
    ]
    prefix = "base_model/" if metadata["format"] == "lora" else ""
    folders.extend(prefix + name for name in components)
    if prefix:
        folders.append("base_model")
    files = collect_files(
        root,
        names={
            "advai_metadata.json",
            "model_index.json",
            "scheduler_config.json",
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.safetensors.index.json",
            "pytorch_lora_weights.safetensors",
            "preprocessor_config.json",
        },
        directories=folders,
    )
    body = f"""Latent text-to-image diffusion with a Stable Diffusion pipeline.
Export format: `{metadata["format"]}`. Full training updates only the UNet;
LoRA training updates only UNet adapters. VAE and text encoder are frozen.

```python
from advai_notebooks.diffusion.model import DiffusionModel

model = DiffusionModel.load({repo_id!r}, revision="main", device="cpu")
images = model.generate(["a red square"], seed=123)
```

Full exports also support native `diffusers.StableDiffusionPipeline.from_pretrained`.
LoRA exports include a frozen initial pipeline in `base_model/`; load it and then
call `pipeline.load_lora_weights` on the adapter directory for native inference.
Restore its trained noise schedule from `noise_scheduler/` using DDIMScheduler or
DDPMScheduler according to `advai_metadata.json` (`scheduler.inference`); the bundled
base keeps its original schedule, which may have a different prediction target.
The project facade resolves this bundled base automatically. Pin the publication SHA.
Weights, tokenizer, component configurations and inference/noise schedules are included.
Training data, optimizer/RNG state, training checkpoints and generated images are excluded.
Document the base model, dataset revision, training settings and image-quality evaluation
in a custom English card. Denoising loss alone does not establish generation quality.
"""
    if metadata["source"].get("random_initialization"):
        body += "\nRandom tiny pipeline / synthetic workflow demo; no image-quality claim.\n"
    content = (
        Path(model_card).read_text()
        if model_card
        else card(
            repo_id,
            "diffusion",
            body,
            library="diffusers",
            license=license,
            pipeline_tag="text-to-image",
            tags=[metadata["format"]],
        )
    )
    return publish_files(
        root,
        repo_id,
        "diffusion",
        files,
        content,
        validate_package=lambda p: DiffusionModel.load(p, local_files_only=True),
        **options,
    )
