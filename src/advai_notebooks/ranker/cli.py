from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.cli import (
    experiment_parser,
    inference_arguments,
    print_result,
    read_inputs,
)
from advai_notebooks.common.prepared_data import load_candidates
from advai_notebooks.common.retrieval_data import load_data
from advai_notebooks.common.runtime import resolve_device, setup

from .config import load_config
from .data import load_evaluation_candidates, load_training_samples


def main(argv=None):
    parser, sub = experiment_parser("advai-ranker")
    sub.choices["evaluate"].add_argument(
        "--candidates", help="Override the configured candidate artifact"
    )
    inference = sub.add_parser("rerank")
    inference_arguments(inference)
    args = parser.parse_args(argv)
    if args.command == "rerank":
        from .model import RankerModel

        inputs = read_inputs(args.input)
        if any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(t, str) for t in pair)
            for pair in inputs
        ):
            raise ValueError("rerank expects [query, document] pairs")
        model = RankerModel.load(args.checkpoint, resolve_device(args.device))
        model.check_lengths(inputs)
        write_json(
            args.output, {"pairs": inputs, "scores": model.score(inputs), "score_type": "raw_logit"}
        )
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
            load_evaluation_candidates(cfg, data)
            result = {
                "kind": "ranker",
                "data": data.summary(),
                "training_rows": len(rows),
                "samples_fingerprint": fingerprint,
            }
        else:
            from .evaluate import evaluate
            from .model import RankerModel

            candidates = load_candidates(
                cfg.path(args.candidates or cfg.data.candidates), data, (args.split,)
            )[args.split]
            report, rankings = evaluate(
                RankerModel.load(args.checkpoint, setup(cfg)), data, cfg, args.split, candidates
            )
            result = {"report": report, "rankings": rankings}
            write_json(args.output, result)
    print_result(result)
