"""Command-line entry points shared with notebooks."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Decision-model validation, fine-tuning and inference"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument(
            "--root", help="Resolve data/model/output paths relative to this directory"
        )
        if name == "run":
            command.add_argument("--resume", help="Original experiment's output_dir/latest")
        if name == "evaluate":
            command.add_argument("--model", required=True, help="Exported model directory")
            command.add_argument(
                "--split", choices=["validation", "calibration", "test"], default="test"
            )
    predict = commands.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--task", required=True)
    predict.add_argument(
        "--state", required=True, help="Text, or a JSON-encoded string/object/list"
    )
    predict.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    predict.add_argument("--min-confidence", type=float)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .pipeline import run_experiment

            report = run_experiment(args.config, args.root, args.resume)
            result = {
                "export_dir": report["export_dir"],
                "metrics": report["fine_tuned_calibrated"] or report["fine_tuned"],
            }
        elif args.command == "predict":
            from advai_notebooks.common.runtime import resolve_device

            from .model import DecisionModel

            device = resolve_device(args.device)
            model = DecisionModel.load(args.model, device)
            try:
                state = json.loads(args.state)
            except json.JSONDecodeError:
                state = args.state
            if not isinstance(state, (str, dict, list)) or not state:
                raise ValueError("state must be nonempty text, object or list")
            result = model.system_one(state, task=args.task, min_confidence=args.min_confidence)
        else:
            from .config import load_config
            from .data import load_data, prepare_training_items, summary

            cfg = load_config(args.config, args.root)
            records, tasks = load_data(cfg.path(cfg.data.path), cfg.path(cfg.data.tasks))
            if args.command == "validate":
                result = summary(records, tasks)
            else:
                from .evaluate import evaluate
                from .model import DecisionModel
                from .runtime import setup

                model = DecisionModel.load(cfg.path(args.model), setup(cfg))
                items = prepare_training_items(
                    [r for r in records if r["split"] == args.split],
                    tasks,
                    model,
                    cfg.data.reject_truncation,
                )
                result, _ = evaluate(model, {args.split: items}, cfg, args.split)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
