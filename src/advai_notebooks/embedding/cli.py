from pathlib import Path

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.cli import (
    experiment_parser,
    inference_arguments,
    print_result,
    read_inputs,
)
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.common.runtime import resolve_device, setup

from .config import load_config
from .data import load_training_samples


def main(argv=None):
    parser, sub = experiment_parser("advai-embedding")
    inference = sub.add_parser("encode")
    inference_arguments(inference)
    inference.add_argument("--role", choices=("query", "document"), required=True)
    args = parser.parse_args(argv)
    if args.command == "encode":
        import numpy as np

        from .model import EmbeddingModel

        inputs = read_inputs(args.input)
        if any(not isinstance(text, str) for text in inputs):
            raise ValueError("encode expects strings")
        model = EmbeddingModel.load(args.checkpoint, resolve_device(args.device))
        model.check_lengths(inputs, args.role)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, model.encode(inputs, args.role).numpy())
        return
    cfg = load_config(args.config, args.root)
    if args.command == "run":
        from .pipeline import run

        result = run(cfg, resume=args.resume)
    else:
        data = load_data(cfg)
        if args.command == "validate":
            for split in ("train", "validation", "test"):
                data.query_ids(split)
            rows, fingerprint = load_training_samples(cfg, data)
            result = {
                "kind": "embedding",
                "data": data.summary(),
                "training_rows": len(rows),
                "samples_fingerprint": fingerprint,
            }
        else:
            from .evaluate import evaluate
            from .model import EmbeddingModel

            report, rankings = evaluate(
                EmbeddingModel.load(args.checkpoint, setup(cfg)), data, cfg, args.split
            )
            result = {"report": report, "rankings": rankings}
            write_json(args.output, result)
    print_result(result)
