"""Compare both retrievers and both rerankers on the same test corpus/qrels."""

from pathlib import Path

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.retrieval import rerank, retrieve
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.common.runtime import setup
from advai_notebooks.embedding.metrics import evaluate_rankings as embedding_metrics
from advai_notebooks.embedding.model import EmbeddingModel
from advai_notebooks.ranker.metrics import evaluate_rankings as ranker_metrics
from advai_notebooks.ranker.model import RankerModel


def compare(cfg, embedding_run, ranker_run, output=None):
    device, data = setup(cfg), load_data(cfg)
    rows = []
    for retrieval_name, checkpoint in (("baseline", "baseline"), ("finetuned", "export")):
        embedding = EmbeddingModel.load(Path(embedding_run) / checkpoint, device)
        if cfg.data.reject_truncation:
            embedding.check_lengths(
                [data.queries[q]["text"] for q in data.query_ids("test")], "query"
            )
            embedding.check_lengths([d["text"] for d in data.corpus.values()], "document")
        candidates = retrieve(
            embedding,
            data,
            data.query_ids("test"),
            cfg.retrieval.top_k,
            cfg.retrieval.corpus_chunk_size,
            cfg.retrieval.query_chunk_size,
            batch_size=cfg.training.batch_size,
        )
        del embedding
        rows.append(
            {
                "embedding": retrieval_name,
                "ranker": "none",
                **embedding_metrics(candidates, data, "test", cfg.evaluation.cutoffs)["metrics"],
            }
        )
        for ranker_name, ranker_checkpoint in (("baseline", "baseline"), ("finetuned", "export")):
            ranker = RankerModel.load(Path(ranker_run) / ranker_checkpoint, device)
            rankings = rerank(
                ranker, data, candidates, cfg.training.batch_size, cfg.data.reject_truncation
            )
            rows.append(
                {
                    "embedding": retrieval_name,
                    "ranker": ranker_name,
                    **ranker_metrics(rankings, data, "test", cfg.evaluation.cutoffs)["metrics"],
                }
            )
            del ranker
    result = {
        "data_fingerprint": data.fingerprint,
        "candidate_top_k": cfg.retrieval.top_k,
        "rows": rows,
    }
    if output:
        write_json(output, result)
    return result
