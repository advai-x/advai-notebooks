"""Train a ranker using prepared pairs and externally supplied candidates."""

import json

from advai_notebooks.common.artifacts import digest, write_json
from advai_notebooks.common.checkpoint import load_resume_state
from advai_notebooks.common.experiment import record_environment, run_signature
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.common.runtime import setup

from .data import load_evaluation_candidates, load_training_samples
from .evaluate import evaluate
from .metrics import evaluate_rankings
from .model import RankerModel
from .train import train_model


def run(cfg, resume=None):
    device, data = setup(cfg), load_data(cfg)
    for split in ("train", "validation", "test"):
        data.query_ids(split)
    output = cfg.path(cfg.output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(f"Output directory is not empty: {output}; choose a new output_dir")
    pairs, samples_fingerprint = load_training_samples(cfg, data)
    candidates = load_evaluation_candidates(cfg, data)
    signature = run_signature(
        cfg,
        {
            "data": data.fingerprint,
            "samples": samples_fingerprint,
            "candidates": digest(candidates),
        },
        device,
    )
    resume_state = load_resume_state(cfg, resume, signature) if resume is not None else None
    model = (
        RankerModel.load(output / "baseline", device)
        if resume_state
        else RankerModel(cfg.model, device, cfg.root)
    )
    model.settings = cfg.model
    if resume_state:
        model.model.load_state_dict(resume_state["model_weights"])
    if cfg.data.reject_truncation:
        model.check_lengths(
            [(data.queries[r["query_id"]]["text"], data.corpus[r["doc_id"]]["text"]) for r in pairs]
        )
    output.mkdir(parents=True, exist_ok=True)
    if resume_state is None:
        record_environment(output, cfg, model)
        write_json(output / "config.json", cfg.to_dict())
        write_json(output / "data_summary.json", data.summary())
        write_json(output / "training_pairs.json", pairs)
        write_json(
            output / "candidates.json",
            {
                "source": {"candidates_file": str(cfg.path(cfg.data.candidates))},
                "schema_version": 1,
                "data_fingerprint": data.fingerprint,
                "splits": candidates,
            },
        )

        baseline, baseline_rankings = evaluate(model, data, cfg, "test", candidates["test"])
        write_json(output / "baseline_test.json", baseline)
        write_json(output / "baseline_test_rankings.json", baseline_rankings)
    else:
        baseline = json.loads((output / "baseline_test.json").read_text())
        baseline_rankings = json.loads((output / "baseline_test_rankings.json").read_text())
    training = train_model(
        model, data, cfg, pairs, candidates["validation"], output, signature, resume_state
    )
    model = None
    selected = RankerModel.load(output / "best", device)
    selected.save(output / "export")
    final, final_rankings = evaluate(selected, data, cfg, "test", candidates["test"])
    write_json(
        output / "test_rankings.json", {"baseline": baseline_rankings, "finetuned": final_rankings}
    )
    report = {
        "kind": "ranker",
        "data": data.summary(),
        "training": training,
        "training_samples_fingerprint": samples_fingerprint,
        "baseline_test": baseline,
        "finetuned_test": final,
        "retrieval_test": evaluate_rankings(
            candidates["test"], data, "test", cfg.evaluation.cutoffs
        ),
        "export": str(output / "export"),
        "device": str(device),
        "note": "Baseline and finetuned ranker use identical candidates. Missing positives are not injected.",
    }
    write_json(output / "report.json", report)
    return report
