"""Native Decision publication preserves tasks and calibrated inference metadata."""

import json
from pathlib import Path

from advai_notebooks.common.hub import card, collect_files, publish_files

from .model import DecisionModel


def publish(export_dir, repo_id, *, license=None, model_card=None, **options):
    root = Path(export_dir).resolve()
    model = DecisionModel.load(root, device="cpu")
    files = collect_files(
        root,
        names={"decision_config.json", "source.json", "tasks.json", "calibration.json"},
        directories=("", "encoder", "tokenizer"),
    )
    task = next(iter(model.task_definitions)) if model.task_definitions else None
    call = (
        f'model.system_one("Customer message", task={task!r})'
        if task
        else ("model.system_one(requests)  # A list of label-free DecisionRequest objects")
    )
    body = f"""Native encoder + option scorer supporting choice, score and noul.
Choice returns an option; score returns an expected ordered level;
noul returns the probability of true. All include probabilities and confidence.
Exported temperature calibration is applied automatically by `system_one`.

Install this project's `decision` extra, then load the published repository:

```python
from advai_notebooks.decision.model import DecisionModel

model = DecisionModel.load({repo_id!r}, revision="main", device="cpu")
answers = {call}
```

Use the project's loader: plain Transformers `AutoModel` does not reconstruct
the native decision head, input encoding or typed decoder.
Pin the publication commit SHA in `revision` for reproducible deployment.
Task definitions (when present) and calibration metadata are included.
Training states, labels and optimizer state are not part of this package.
"""
    if model.source.get("random_initialization"):
        body += "\nRandom tiny model / synthetic smoke artifact; not a domain benchmark.\n"
    report_path = root.parent / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if Path(report.get("export_dir", "")).resolve() == root:
            metrics = report["fine_tuned_calibrated"] or report["fine_tuned"]
            body += f"\nSelected checkpoint: epoch {report['best_epoch']}.\n"
            body += (
                "\nHeld-out test metrics:\n```json\n"
                + json.dumps(metrics["overall"], indent=2)
                + "\n```\n"
            )
    body += "\nDocument domain training data and evaluation in a custom model card.\n"
    content = (
        Path(model_card).read_text()
        if model_card
        else card(
            repo_id,
            "decision",
            body,
            library="pytorch",
            license=license,
            tags=("choice", "score", "noul"),
        )
    )
    del model
    return publish_files(
        root, repo_id, "decision", files, content, validate_package=DecisionModel.load, **options
    )
