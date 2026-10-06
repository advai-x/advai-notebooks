"""Base publication owns its native full/adapter contract and English card."""

import json
from pathlib import Path

from advai_notebooks.common.hub import card, collect_files, publish_files

from .model import BaseModel


def publish(export_dir, repo_id, *, license=None, model_card=None, **options):
    root = Path(export_dir).resolve()
    metadata = json.loads((root / "advai_metadata.json").read_text())
    if metadata["kind"] != "llm/base":
        raise ValueError("Expected a base inference export")
    directories = [""]
    base = metadata.get("base_model")
    if base and base["kind"] == "bundled":
        directories.append(base["path"])
    files = collect_files(
        root,
        names={
            "advai_metadata.json",
            "generation_config.json",
            "adapter_config.json",
            "adapter_model.safetensors",
            "chat_template.jinja",
        },
        directories=directories,
    )
    template_dir = root / "chat_templates"
    if template_dir.is_dir():
        templates = [p.name for p in template_dir.iterdir() if p.suffix == ".jinja"]
        files.extend(collect_files(root, names=templates, directories=["chat_templates"]))
    body = """Continued pre-training / domain adaptation with next-token causal language modeling.

Install this project's `llm-base` extra and load the versioned export:

```python
from advai_notebooks.llm.base.model import BaseModel

model = BaseModel.load(REPO_ID, revision="main", device="cpu")
outputs = model.generate(["the sky is"], max_new_tokens=8)
```

This package has export format `FORMAT`.
Full models also support `transformers.AutoModelForCausalLM.from_pretrained`.
Adapters require PEFT and the exact recorded base revision. A bundled base is
resolved relative to the export directory by this project's model facade.
The tokenizer and chat template are exported with the model.
Pin the publication commit SHA for reproducible deployment.
Raw training data, optimizer state and training checkpoints are excluded.
Document domain data, training configuration and evaluation in a custom English card.
""".replace("REPO_ID", repr(repo_id)).replace("FORMAT", metadata["format"])
    if metadata["source"]["id"] == "tiny":
        body += "\nRandom tiny decoder / synthetic smoke artifact; no domain-quality claim.\n"
    report_path = root.parent / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if Path(report.get("export", "")).resolve() == root:
            body += f"\nSelected checkpoint: epoch {report['training']['best_epoch']}.\n"
            body += (
                "\nHeld-out metrics:\n```json\n"
                + json.dumps(report["finetuned_test"]["metrics"], indent=2)
                + "\n```\n"
            )
    content = (
        Path(model_card).read_text()
        if model_card
        else card(
            repo_id,
            "llm-base",
            body,
            library="transformers",
            license=license,
            pipeline_tag="text-generation",
            tags=[metadata["format"]],
        )
    )
    return publish_files(
        root,
        repo_id,
        "llm/base",
        files,
        content,
        validate_package=lambda path: BaseModel.load(path, local_files_only=True),
        **options,
    )
