"""Whitelist-only publication mechanics with injected native package validation."""

from pathlib import Path

from .hub import card, collect_files, publish_files

ASSETS = (
    "advai_metadata.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "adapter_config.json",
    "adapter_model.safetensors",
    "action_head.safetensors",
)


def publish(
    export_dir,
    repo_id,
    kind,
    model_class,
    body,
    *,
    private=True,
    revision="main",
    token=None,
    license=None,
    model_card=None,
    commit_message=None,
    dry_run=False,
    asset_names=(),
    component_dirs=(),
    pipeline_tag=None,
):
    export_dir = Path(export_dir).resolve()
    if not export_dir.is_dir():
        raise FileNotFoundError("Publication requires an existing local inference export")
    model_class.load(export_dir, "cpu")  # Validate kind before staging.
    files = collect_files(
        export_dir,
        names=(*ASSETS, *asset_names),
        directories=("", "encoder", "base_model", *component_dirs),
    )
    for directory in ("chat_templates", "encoder/chat_templates"):
        names = [file.name for file in (export_dir / directory).glob("*.jinja")]
        files.extend(collect_files(export_dir, names=names, directories=(directory,)))
    files = sorted(set(files))
    document = (
        Path(model_card).read_text()
        if model_card
        else card(
            repo_id,
            kind,
            body,
            library="advai-notebooks",
            license=license,
            tags=(kind, "training-guide"),
            pipeline_tag=pipeline_tag,
        )
    )
    return publish_files(
        export_dir,
        repo_id,
        kind,
        files,
        document,
        private=private,
        revision=revision,
        token=token,
        dry_run=dry_run,
        commit_message=commit_message,
        validate_package=lambda p: model_class.load(p, "cpu"),
    )
