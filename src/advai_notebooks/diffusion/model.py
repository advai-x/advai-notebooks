"""Public diffusion facade: components, native full/LoRA persistence and generation."""

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.hub import resolve_export

from .backbones import DiffusionBackbone, tiny_pipeline
from .config import GenerationConfig, ModelConfig, SchedulerConfig, TrainingConfig
from .heads import DenoisingHead
from .schedulers import inference_scheduler, noise_scheduler, scheduler_config


def load_pipeline(path, **options):
    index = json.loads((Path(path) / "model_index.json").read_text())
    if index.get("_class_name") != "StableDiffusionPipeline":
        raise ValueError("Diffusion v1 supports StableDiffusionPipeline, not SDXL/DiT/video")
    return StableDiffusionPipeline.from_pretrained(path, use_safetensors=True, **options)


def pretrained_files():
    """Download only native inference components, excluding duplicate precision/bin weights."""
    files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "feature_extractor/preprocessor_config.json",
    ]
    for component in ("unet", "vae"):
        files.extend(
            f"{component}/{name}"
            for name in (
                "config.json",
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.safetensors.index.json",
                "diffusion_pytorch_model-*.safetensors",
            )
        )
    for component in ("text_encoder", "safety_checker"):
        files.extend(
            f"{component}/{name}"
            for name in (
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
            )
        )
    files.extend(
        f"tokenizer/{name}"
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
    )
    return files


class DiffusionModel:
    def __init__(
        self, settings=None, device="cpu", root=None, training=None, generation=None, scheduler=None
    ):
        self.settings = replace(settings) if settings else ModelConfig()
        self.generation = generation or GenerationConfig()
        self.scheduler_settings = scheduler or SchedulerConfig()
        self.device = torch.device(device)
        self._temporary = None
        self.base_dir = None
        self.method = "full"
        source = self.settings.id
        local = Path(root or ".") / Path(source).expanduser()
        if source == "tiny":
            self.pipeline = tiny_pipeline()
            self.source = {"id": "tiny", "revision": None, "random_initialization": True}
        else:
            if local.is_dir():
                local = local.resolve()
                metadata = local / "advai_metadata.json"
                if metadata.exists() and json.loads(metadata.read_text()).get("format") != "full":
                    raise ValueError("Initialize new training from a full/merged pipeline")
            else:
                if Path(source).expanduser().is_absolute() or source.startswith((".", "~")):
                    raise FileNotFoundError(f"Native pipeline directory does not exist: {source}")
                from huggingface_hub import HfApi, snapshot_download

                sha = HfApi().model_info(source, revision=self.settings.revision).sha
                local = Path(
                    snapshot_download(source, revision=sha, allow_patterns=pretrained_files())
                )
                self.settings.revision = sha
            self.pipeline = load_pipeline(local, local_files_only=True)
            self.source = {
                "id": source,
                "revision": self.settings.revision,
                "random_initialization": False,
            }
            own = local / "advai_metadata.json"
            if own.exists():
                self.source["random_initialization"] = json.loads(own.read_text())["source"].get(
                    "random_initialization", False
                )
        self.noise_scheduler = noise_scheduler(
            self.pipeline.scheduler.config, self.scheduler_settings.prediction_type
        )
        self._components()
        training = training or TrainingConfig()
        if training.method == "lora":
            self._temporary = TemporaryDirectory(prefix="advai-diffusion-base-")
            self.base_dir = Path(self._temporary.name) / "base_model"
            self.pipeline.save_pretrained(self.base_dir, safe_serialization=True)
            self.pipeline.unet.add_adapter(
                LoraConfig(
                    r=training.lora_rank,
                    lora_alpha=training.lora_alpha,
                    lora_dropout=0.0,
                    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                ),
                adapter_name="default",
            )
            self.method = "lora"
        self._trainable(True)
        if training.gradient_checkpointing:
            self.pipeline.unet.enable_gradient_checkpointing()
        self._validate_limits()

    def _components(self):
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.backbone = DiffusionBackbone(self.pipeline)
        self.head = DenoisingHead(self.pipeline.unet)
        self.pipeline.scheduler = inference_scheduler(
            self.noise_scheduler.config, self.scheduler_settings.inference
        )

    def _trainable(self, enabled):
        for module in (self.pipeline.vae, self.pipeline.text_encoder):
            module.requires_grad_(False)
            module.eval()
        for name, parameter in self.pipeline.unet.named_parameters():
            parameter.requires_grad_(enabled and (self.method == "full" or "lora_" in name))
        self.pipeline.unet.eval()

    def _validate_limits(self):
        scale = self.pipeline.vae_scale_factor
        downsample = 2 ** (len(self.pipeline.unet.config.block_out_channels) - 1)
        if (
            self.settings.resolution % 8
            or self.settings.resolution % scale
            or (self.settings.resolution // scale) % downsample
        ):
            raise ValueError("Resolution incompatible with VAE/UNet downsampling")
        if self.generation.num_inference_steps > self.noise_scheduler.config.num_train_timesteps:
            raise ValueError("Inference steps exceed noise schedule length")
        if self.pipeline.unet.config.in_channels != self.pipeline.vae.config.latent_channels:
            raise ValueError("Only standard text-to-image latent channels are supported")

    def validate_prompts(self, prompts):
        if (
            not isinstance(prompts, list)
            or not prompts
            or any(not isinstance(p, str) or not p.strip() for p in prompts)
        ):
            raise ValueError("Prompts must be a nonempty list of nonempty strings")
        maximum = self.pipeline.tokenizer.model_max_length
        if maximum > self.pipeline.text_encoder.config.max_position_embeddings:
            raise ValueError("Tokenizer budget exceeds text encoder context")
        for prompt in prompts:
            if len(self.pipeline.tokenizer(prompt, truncation=False).input_ids) > maximum:
                raise ValueError("Caption/prompt exceeds text encoder context; truncation rejected")

    @classmethod
    def load(cls, path, device="cpu", *, trainable=False, **hub_options):
        root = resolve_export(path, **hub_options)
        meta = json.loads((root / "advai_metadata.json").read_text())
        if meta.get("schema_version") != 1 or meta.get("kind") != "diffusion":
            raise ValueError("Expected a versioned diffusion export")
        obj = cls.__new__(cls)
        obj.settings = ModelConfig(**meta["model"])
        obj.generation = GenerationConfig(**meta["generation"])
        obj.scheduler_settings = SchedulerConfig(**meta["scheduler"])
        obj.source = meta["source"]
        obj.device = torch.device(device)
        obj.method = meta["format"]
        obj._temporary = None
        obj.base_dir = None
        if obj.method == "full":
            obj.pipeline = load_pipeline(root, local_files_only=True)
        elif obj.method == "lora":
            relative = Path(meta["base_model"]["path"])
            obj.base_dir = (root / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not obj.base_dir.is_relative_to(root)
            ):
                raise ValueError("Bundled base escapes export directory")
            obj.pipeline = load_pipeline(obj.base_dir, local_files_only=True)
            obj.pipeline.load_lora_weights(root, adapter_name="default", local_files_only=True)
        else:
            raise ValueError("Unsupported diffusion export format")
        obj.noise_scheduler = noise_scheduler(meta["noise_scheduler"])
        obj._components()
        obj._trainable(trainable)
        obj._validate_limits()
        return obj

    def save(self, path, *, merged=False):
        path = Path(path).resolve()
        if path.exists() and any(path.iterdir()):
            raise ValueError("Export directory must be empty; choose a new path")
        path.mkdir(parents=True, exist_ok=True)
        if merged and self.method == "lora":
            with TemporaryDirectory(prefix="advai-diffusion-merge-") as directory:
                self.save(Path(directory) / "adapter")
                clone = type(self).load(Path(directory) / "adapter", self.device)
                clone.pipeline.fuse_lora()
                clone.pipeline.unload_lora_weights()
                clone.method = "full"
                return clone.save(path)
        if self.method == "full":
            self.pipeline.save_pretrained(path, safe_serialization=True)
        else:
            shutil.copytree(self.base_dir, path / "base_model")
            metadata = self.pipeline.unet.peft_config["default"].to_dict()
            metadata["target_modules"] = sorted(metadata["target_modules"])
            self.pipeline.save_lora_weights(
                path,
                unet_lora_layers=convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(self.pipeline.unet, adapter_name="default")
                ),
                unet_lora_adapter_metadata=metadata,
                safe_serialization=True,
            )
        self.noise_scheduler.save_pretrained(path / "noise_scheduler")
        write_json(
            path / "advai_metadata.json",
            {
                "schema_version": 1,
                "kind": "diffusion",
                "format": self.method,
                "model": asdict(self.settings),
                "generation": asdict(self.generation),
                "scheduler": asdict(self.scheduler_settings),
                "source": self.source,
                "noise_scheduler": scheduler_config(self.noise_scheduler.config),
                "base_model": {"kind": "bundled", "path": "base_model"}
                if self.method == "lora"
                else None,
            },
        )
        return path

    def configure(self, settings, generation, scheduler):
        self.settings.resolution = settings.resolution
        self.generation = generation
        self.scheduler_settings = scheduler
        if scheduler.prediction_type is not None and (
            scheduler.prediction_type != self.noise_scheduler.config.prediction_type
        ):
            raise ValueError("Evaluation objective differs from the trained prediction_type")
        self._validate_limits()

    def generate(self, prompts, *, seed=None, num_inference_steps=None, guidance_scale=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.validate_prompts(prompts)
        seed = self.generation.seed if seed is None else seed
        steps = (
            self.generation.num_inference_steps
            if num_inference_steps is None
            else num_inference_steps
        )
        guidance = self.generation.guidance_scale if guidance_scale is None else guidance_scale
        from advai_notebooks.common.configuration import positive_int, positive_number

        positive_int("num_inference_steps", steps)
        positive_number("guidance_scale", guidance, True)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if steps > self.noise_scheduler.config.num_train_timesteps:
            raise ValueError("Inference steps exceed noise schedule length")
        self.pipeline.unet.eval()
        self.pipeline.scheduler = inference_scheduler(
            self.noise_scheduler.config, self.scheduler_settings.inference
        )
        # Per-image CPU generators isolate inference randomness from training RNG.
        generators = [
            torch.Generator(device="cpu").manual_seed((seed + i) % 2**32)
            for i in range(len(prompts))
        ]
        return self.pipeline(
            prompts,
            height=self.settings.resolution,
            width=self.settings.resolution,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generators,
        ).images

    def save_images(self, prompts, path, **options):
        images = self.generate(prompts, **options)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        records = []
        for i, image in enumerate(images):
            name = f"image-{i:04d}.png"
            if (path / name).exists():
                raise ValueError("Generated image already exists; choose a fresh output directory")
            image.save(path / name)
            records.append(
                {
                    "prompt": prompts if isinstance(prompts, str) else prompts[i],
                    "path": str((path / name).resolve()),
                    "size": list(image.size),
                }
            )
        write_json(path / "images.json", {"kind": "diffusion", "images": records})
        return {"kind": "diffusion", "images": records}
