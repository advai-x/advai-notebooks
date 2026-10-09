"""Independent diffusion CLI; generate/export outputs are directories."""

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
from .model import DiffusionModel


def main(argv=None):
    parser, sub = experiment_parser("advai-diffusion")
    publish_parser(sub)
    generate = sub.add_parser("generate")
    inference_arguments(generate)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--num-inference-steps", type=int)
    generate.add_argument("--guidance-scale", type=float)
    export = sub.add_parser("export")
    inference_arguments(export)
    export._option_string_actions["--input"].required = False
    export.add_argument("--merged", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "publish":
        from .publish import publish

        publish_result(args, publish)
        return
    if args.command in {"generate", "export"}:
        model = DiffusionModel.load(
            args.checkpoint, resolve_device(args.device), **hub_options(args)
        )
        result = (
            {"export": str(model.save(args.output, merged=args.merged))}
            if args.command == "export"
            else (
                model.save_images(
                    read_inputs(args.input),
                    args.output,
                    seed=args.seed,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )
            )
        )
    else:
        cfg = load_config(args.config, args.root)
        if args.command == "run":
            from .pipeline import run

            result = run(cfg, args.resume)
        else:
            rows, summary = load_data(cfg)
            if args.command == "validate":
                model = DiffusionModel(
                    cfg.model, setup(cfg), cfg.root, cfg.training, cfg.generation, cfg.scheduler
                )
                prepared = prepare_data(rows, model, cfg)
                result = {
                    "kind": cfg.kind,
                    "data": summary,
                    "prepared_rows": {k: len(v) for k, v in prepared.items()},
                }
            else:
                from .evaluate import evaluate

                model = DiffusionModel.load(args.checkpoint, setup(cfg), **hub_options(args))
                model.configure(cfg.model, cfg.generation, cfg.scheduler)
                result = evaluate(model, prepare_data(rows, model, cfg), cfg, args.split)
                write_json(args.output, result)
    print_result(result)
