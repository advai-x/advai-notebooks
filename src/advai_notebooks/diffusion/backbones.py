"""Native Diffusers components; no imports from other model tasks."""

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import CLIPTextConfig, CLIPTextModel, PreTrainedTokenizerFast


def tiny_pipeline():
    words = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "a",
        "red",
        "green",
        "blue",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "pink",
        "square",
        "pattern",
        "bright",
        "dark",
    ]
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=16,
    )
    encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(words),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            attention_dropout=0.0,
        )
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(16,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=16,
    )
    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=4,
    )
    return StableDiffusionPipeline(
        vae=vae,
        text_encoder=encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDPMScheduler(
            num_train_timesteps=32, beta_schedule="scaled_linear", clip_sample=False, steps_offset=1
        ),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


class DiffusionBackbone:
    def __init__(self, pipeline):
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer

    def text_features(self, captions):
        encoded = self.tokenizer(
            captions,
            padding="max_length",
            truncation=False,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.unet.device)
        options = (
            {"attention_mask": encoded.attention_mask}
            if getattr(self.text_encoder.config, "use_attention_mask", False)
            else {}
        )
        with torch.no_grad():
            return self.text_encoder(encoded.input_ids, **options)[0]

    def latents(self, pixels, *, deterministic=False):
        with torch.no_grad():
            distribution = self.vae.encode(pixels).latent_dist
            return (distribution.mode() if deterministic else distribution.sample()) * (
                self.vae.config.scaling_factor
            )
