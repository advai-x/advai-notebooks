"""Train an embedding model using independently prepared contrastive samples."""

import json

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.checkpoint import load_resume_state
from advai_notebooks.common.experiment import record_environment, run_signature
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.common.runtime import setup

from .data import load_training_samples
from .evaluate import evaluate
from .model import EmbeddingModel
from .sampling import embedding_batches
from .train import train_model


def run(cfg, resume=None):
    device, data = setup(cfg), load_data(cfg)
    for split in ("train", "validation", "test"):
        data.query_ids(split)
    output = cfg.path(cfg.output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(f"Output directory is not empty: {output}; choose a new output_dir")
    rows, samples_fingerprint = load_training_samples(cfg, data)
    embedding_batches(rows, data, cfg.training.batch_size, cfg.training.seed)
    signature = run_signature(
        cfg, {"data": data.fingerprint, "samples": samples_fingerprint}, device
    )
    resume_state = load_resume_state(cfg, resume, signature) if resume is not None else None
    model = (
        EmbeddingModel.load(output / "baseline", device)
        if resume_state
        else EmbeddingModel(cfg.model, device, cfg.root)
    )
    model.settings = cfg.model
    if resume_state:
        model.model.load_state_dict(resume_state["model_weights"])
    if cfg.data.reject_truncation:
        model.check_lengths([q["text"] for q in data.queries.values()], "query")
        model.check_lengths([d["text"] for d in data.corpus.values()], "document")
    output.mkdir(parents=True, exist_ok=True)
    if resume_state is None:
        record_environment(output, cfg, model)
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "data_summary.json", data.summary())
        write_json(
            output / "training_samples.json",
            {
                "schema_version": 1,
                "kind": "embedding",
                "data_fingerprint": data.fingerprint,
                "rows": rows,
            },
        )
        baseline, baseline_rankings = evaluate(model, data, cfg, "test")
        write_json(output / "baseline_test.json", baseline)
        write_json(output / "baseline_test_rankings.json", baseline_rankings)
    else:
        baseline = json.loads((output / "baseline_test.json").read_text())
        baseline_rankings = json.loads((output / "baseline_test_rankings.json").read_text())
    training = train_model(model, data, cfg, rows, output, signature, resume_state)
    model = None
    selected = EmbeddingModel.load(output / "best", device)
    selected.save(output / "export")
    final, final_rankings = evaluate(selected, data, cfg, "test")
    write_json(
        output / "test_rankings.json", {"baseline": baseline_rankings, "finetuned": final_rankings}
    )
    report = {
        "kind": "embedding",
        "data": data.summary(),
        "training": training,
        "training_samples_fingerprint": samples_fingerprint,
        "baseline_test": baseline,
        "finetuned_test": final,
        "export": str(output / "export"),
        "device": str(device),
        "note": "Validation selects the checkpoint, including epoch zero. Test is reported only.",
    }
    write_json(output / "report.json", report)
    return report
