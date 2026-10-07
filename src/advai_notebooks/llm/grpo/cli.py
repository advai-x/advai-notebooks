"""Standalone grpo CLI: validation, training/resume, evaluation, generate/chat and export."""

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

from .config import load_config
from .data import load_data, prepare_data
from .model import GRPOModel


def main(argv=None):
    parser, sub = experiment_parser("advai-llm-grpo")
    publish_parser(sub)
    for name in ("generate", "chat", "export"):
        item = sub.add_parser(name)
        inference_arguments(item)
        if name == "export":
            item._option_string_actions["--input"].required = False
            item.add_argument("--merged", action="store_true")
        else:
            item.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args(argv)
    if args.command == "publish":
        from .publish import publish

        publish_result(args, publish)
        return
    if args.command in {"generate", "chat", "export"}:
        model = GRPOModel.load(args.checkpoint, resolve_device(args.device), **hub_options(args))
        if args.command == "export":
            result = {"export": str(model.save(args.output, merged=args.merged))}
        else:
            inputs = read_inputs(args.input)
            outputs = (
                model.generate(inputs, max_new_tokens=args.max_new_tokens)
                if args.command == "generate"
                else [model.chat(m, max_new_tokens=args.max_new_tokens) for m in inputs]
            )
            result = {"outputs": outputs}
            write_json(args.output, result)
        print_result(result)
        return
    cfg = load_config(args.config, args.root)
    if args.command == "run":
        from .pipeline import run

        result = run(cfg, args.resume)
    else:
        rows, summary = load_data(cfg)
        device = setup(cfg)
        if args.command == "validate":
            model = GRPOModel(cfg.model, device, cfg.root, generation=cfg.generation)
            prepared = prepare_data(rows, model, cfg)
            result = {
                "kind": cfg.kind,
                "data": summary,
                "prepared_rows": {split: len(values) for split, values in prepared.items()},
            }
        else:
            from .evaluate import evaluate

            model = GRPOModel.load(args.checkpoint, device, **hub_options(args))
            model.configure(cfg.model, cfg.generation)
            result = evaluate(model, prepare_data(rows, model, cfg), cfg, args.split)
            write_json(args.output, result)
    print_result(result)
