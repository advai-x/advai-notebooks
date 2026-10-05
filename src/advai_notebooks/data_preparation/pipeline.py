"""Explicit offline stage writing files consumed by independent trainers."""

from pathlib import Path

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.retrieval import retrieve
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.common.runtime import setup

from .mining import mine_negatives, ranker_rows


def load_encoder(cfg, device):
    # Import the optional encoder only in preparation, never inside ranker training.
    from advai_notebooks.embedding.model import EmbeddingModel

    source = cfg.model_path(cfg.model.id)
    if (Path(source) / "advai_metadata.json").is_file():
        return EmbeddingModel.load(source, device)
    return EmbeddingModel(cfg.model, device, cfg.root)


def prepare(cfg):
    device, data = setup(cfg), load_data(cfg)
    for split in ("train", "validation", "test"):
        data.query_ids(split)
    output = cfg.path(cfg.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    model = None
    if "embedding" in (cfg.mining.strategy, cfg.mining.candidate_strategy):
        model = load_encoder(cfg, device)
        if cfg.data.reject_truncation:
            model.check_lengths([q["text"] for q in data.queries.values()], "query")
            model.check_lengths([d["text"] for d in data.corpus.values()], "document")
    mined = mine_negatives(model, data, cfg)
    pairs = ranker_rows(mined)
    candidates = {}
    for split in ("validation", "test"):
        if cfg.mining.candidate_strategy == "all_corpus":
            candidates[split] = {
                q: [{"doc_id": d, "score": 0.0} for d in sorted(data.corpus)]
                for q in data.query_ids(split)
            }
        else:
            candidates[split] = retrieve(
                model,
                data,
                data.query_ids(split),
                cfg.retrieval.top_k,
                cfg.retrieval.corpus_chunk_size,
                cfg.retrieval.query_chunk_size,
                batch_size=cfg.training.batch_size,
            )
    source = {"mining": vars(cfg.mining), "encoder": vars(model.settings) if model else None}
    output.mkdir(parents=True, exist_ok=True)
    for kind, rows in (("embedding", mined), ("ranker", pairs)):
        write_json(
            output / f"{kind}_samples.json",
            {
                "schema_version": 1,
                "kind": kind,
                "data_fingerprint": data.fingerprint,
                "source": source,
                "rows": rows,
            },
        )
    write_json(
        output / "candidates.json",
        {
            "schema_version": 1,
            "data_fingerprint": data.fingerprint,
            "source": source,
            "splits": candidates,
        },
    )
    write_json(output / "config.json", cfg.to_dict())
    result = {
        "data": data.summary(),
        "embedding_samples": str(output / "embedding_samples.json"),
        "ranker_samples": str(output / "ranker_samples.json"),
        "candidates": str(output / "candidates.json"),
    }
    write_json(output / "manifest.json", result)
    return result
