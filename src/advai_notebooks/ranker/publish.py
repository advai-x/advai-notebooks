"""Ranker publication owns its native file contract and scoring documentation."""

import json
from pathlib import Path

from advai_notebooks.common.hub import card, collect_files, publish_files

from .model import RankerModel


def publish(export_dir, repo_id, *, license=None, model_card=None, **options):
    root = Path(export_dir).resolve()
    metadata = json.loads((root / "advai_metadata.json").read_text())
    if metadata["kind"] != "ranker":
        raise ValueError("Expected a ranker inference export")
    modules = json.loads((root / "modules.json").read_text())
    files = collect_files(
        root, names={"advai_metadata.json"}, directories=["", *(m["path"] for m in modules)]
    )
    body = """Single-logit cross-encoder for query/document relevance.
Scores are raw logits; higher values indicate greater relevance.
Query and document prefixes are applied once by the task facade.

Install this project's `ranker` extra, then load the published repository:

```python
from advai_notebooks.ranker.model import RankerModel

model = RankerModel.load(REPO_ID, revision="main", device="cpu")
scores = model.score([("reset password", "account login password reset")])
```

The native export also supports `sentence_transformers.CrossEncoder`.
Preserve prefixes and use an identity activation when comparing raw logits.
Pin the publication commit SHA in `revision` for reproducible deployment.
Training pairs, evaluation candidates and optimizer state are not uploaded.
""".replace("REPO_ID", repr(repo_id))
    tiny = metadata["model"]["id"] == "tiny"
    report_path = root.parent / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if Path(report.get("export", "")).resolve() == root:
            body += f"\nSelected checkpoint: epoch {report['training']['best_epoch']}.\n"
            body += (
                "\nHeld-out test metrics:\n```json\n"
                + json.dumps(report["finetuned_test"]["metrics"], indent=2)
                + "\n```\n"
            )
            environment = root.parent / "environment.json"
            if environment.is_file():
                tiny = json.loads(environment.read_text()).get("tiny_model", tiny)
    if tiny:
        body += "\nRandom tiny model / synthetic smoke artifact; not a domain benchmark.\n"
    body += "\nDocument domain training data and evaluation in a custom model card.\n"
    content = (
        Path(model_card).read_text()
        if model_card
        else card(
            repo_id,
            "ranker",
            body,
            library="sentence-transformers",
            license=license,
            pipeline_tag="text-classification",
        )
    )
    return publish_files(
        root, repo_id, "ranker", files, content, validate_package=RankerModel.load, **options
    )
