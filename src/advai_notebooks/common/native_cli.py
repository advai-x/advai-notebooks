"""CLI mechanics; task owners supply all model, data and inference operations."""

from .artifacts import write_json
from .cli import (
    experiment_parser,
    hub_options,
    inference_arguments,
    print_result,
    publish_parser,
    publish_result,
    read_inputs,
)
from .runtime import resolve_device, setup


def main(
    argv,
    *,
    kind,
    load_config,
    model_class,
    load_data,
    prepare_data,
    run,
    evaluate,
    publish,
    inference,
    infer_one,
    supports_merge=False,
):
    parser, sub = experiment_parser(f"advai-{kind}")
    publish_parser(sub)
    inference_arguments(sub.add_parser(inference))
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    from .cli import hub_arguments

    hub_arguments(export)
    if supports_merge:
        export.add_argument("--merged", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "publish":
        publish_result(args, publish)
        return
    if args.command in {inference, "export"}:
        model = model_class.load(args.checkpoint, resolve_device(args.device), **hub_options(args))
        if args.command == "export":
            from pathlib import Path

            destination = Path(args.output)
            if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
                raise ValueError("Export destination must be a new or empty directory")
            result = {
                "export": str(
                    model.save(args.output, **({"merged": args.merged} if supports_merge else {}))
                )
            }
        else:
            result = [infer_one(model, row) for row in read_inputs(args.input)]
            write_json(args.output, result)
    else:
        cfg = load_config(args.config, args.root)
        if args.command == "run":
            result = run(cfg, args.resume)
        else:
            rows, summary = load_data(cfg)
            device = setup(cfg)
            if args.command == "validate":
                model = model_class(cfg.model, device, cfg.root, cfg.training)
                prepared = prepare_data(rows, model, cfg, fit=True)
                result = {
                    "kind": kind,
                    "data": summary,
                    "prepared_rows": {k: len(v) for k, v in prepared.items()},
                }
            else:
                model = model_class.load(args.checkpoint, device, **hub_options(args))
                result = evaluate(model, prepare_data(rows, model, cfg, fit=False), cfg, args.split)
                write_json(args.output, result)
    print_result(result)
