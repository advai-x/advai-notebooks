# VLM — Vision-Language Models

Status: guide outline and directory scaffold. Training, inference, CLI commands,
dependency extras and runnable notebooks are not implemented yet.

This track will cover fine-tuning models that combine visual inputs with language,
including image question answering, image captioning and multimodal conversations.
Video support will depend on the selected backend and its processor.

## Planned workflow

1. Select a compatible checkpoint and preserve its processor, image tokens and chat template.
2. Validate media references, conversation roles and grouped train/validation/test splits.
3. Construct multimodal batches with model-specific visual preprocessing and label masking.
4. Train on assistant responses, with explicit choices about frozen vision components,
   projector updates, full language-model updates or LoRA.
5. Evaluate on held-out prompts using task-appropriate answer metrics and reviewed examples.
6. Export and reload weights together with the processor and multimodal configuration.

## Planned data contract

Records will identify the sample, source group, split, media references and conversation.
The backend must define image ordering and how media placeholders align with messages.
Video records also need timestamps or a reproducible frame-sampling policy.

Related images, video frames and conversations from the same source should remain in
one split. Data validation must detect missing media and unsupported input formats.
The schema will be finalized and tested with the first executable backend.

## Implementation locations

| Area | Directory |
| --- | --- |
| Notebook | [notebooks/vlm](../notebooks/vlm/README.md) |
| Configuration | [configs/vlm](../configs/vlm/README.md) |
| Python package | [src/advai_notebooks/vlm](../src/advai_notebooks/vlm/) |
| Sample data | [data/sample/vlm](../data/sample/vlm/README.md) |
| Tests | [tests/vlm](../tests/vlm/README.md) |
| Outputs | [outputs/vlm](../outputs/vlm/README.md) |

The first backend should include an offline smoke fixture, native export/reload checks
and separately documented validation on a real checkpoint. Existing text-only LLM
presets do not implement multimodal batching or visual-input supervision.
