"""Standalone instruct command-line entry point."""

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.cli import (
    experiment_parser,
    hub_options,
    inference_arguments,
    print_result,
    publish_parser,
    publish_result,
    read_inputs,
)
from advai_notebooks.common.runtime import resolve_device, setup
from advai_notebooks.llm.moe import describe

from .config import load_config
from .data import load_data, prepare_data
from .model import InstructModel


def main(argv=None):
    parser, sub = experiment_parser("advai-llm-instruct")
    sub.choices["run"]._option_string_actions[
        "--resume"
    ].help = "Complete Trainer checkpoint, or this experiment's output_dir/latest alias"
    publish_parser(sub)
    inference = sub.add_parser("chat")
    inference_arguments(inference)
    inference.add_argument("--max-new-tokens", type=int)
    export = sub.add_parser("export")
    inference_arguments(export)
    export._option_string_actions["--input"].required = False
    export.add_argument("--merged", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "publish":
        from .publish import publish

        publish_result(args, publish)
        return
    if args.command in {"chat", "export"}:
        model = InstructModel.load(
            args.checkpoint, resolve_device(args.device), **hub_options(args)
        )
        if args.command == "export":
            result = {"export": str(model.save(args.output, merged=args.merged))}
        else:
            inputs = read_inputs(args.input)
            result = {
                "outputs": [
                    model.chat(messages, max_new_tokens=args.max_new_tokens) for messages in inputs
                ]
            }
            write_json(args.output, result)
        print_result(result)
        return
    cfg = load_config(args.config, args.root)
    if args.command == "run":
        from .pipeline import run

        result = run(cfg, args.resume)
    else:
        rows, summary = load_data(cfg)
        if args.command == "validate":
            model = InstructModel(cfg.model, setup(cfg), cfg.root, cfg.training, cfg.generation)
            prepared = prepare_data(rows, model, cfg)
            result = {
                "kind": cfg.kind,
                "data": summary,
                "encoded_rows": {name: len(values) for name, values in prepared.items()},
                "supervised_tokens": {
                    name: sum(sum(t != -100 for t in row["labels"][1:]) for row in values)
                    for name, values in prepared.items()
                },
            }
            if describe(model.model):
                result["moe"] = describe(model.model)
        else:
            from .evaluate import evaluate

            model = InstructModel.load(args.checkpoint, setup(cfg), **hub_options(args))
            model.configure(cfg.model, cfg.generation)
            result = evaluate(model, prepare_data(rows, model, cfg), cfg, args.split, raw_rows=rows)
            write_json(args.output, result)
    print_result(result)
