"""VAD lifecycle plus offline detection and safe audio splitting."""

import argparse

from advai_notebooks.common.artifacts import write_json
from advai_notebooks.common.cli import hub_arguments, hub_options, print_result
from advai_notebooks.common.native_cli import main as execute
from advai_notebooks.common.runtime import resolve_device

from .config import load_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .model import VADModel
from .pipeline import run
from .publish import publish


def main(argv=None):
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["-h"], ["--help"]):
        parser = argparse.ArgumentParser(prog="advai-vad", description=__doc__)
        sub = parser.add_subparsers(dest="command")
        for name, description in (
            ("validate", "Validate configuration, audio and annotations"),
            ("run", "Train, resume, select and calibrate"),
            ("evaluate", "Evaluate a held-out split"),
            ("export", "Copy a trusted inference export"),
            ("publish", "Explicit inference-only Hub publication or dry run"),
            ("detect", "Detect speech intervals in one audio file"),
            ("detect-batch", "Detect a JSON list of audio paths"),
            ("segment", "Write bounded chunks and original source offsets"),
        ):
            sub.add_parser(name, help=description)
        parser.print_help()
        return
    if arguments and arguments[0] == "segment":
        parser = argparse.ArgumentParser(prog="advai-vad segment")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--audio", required=True)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--max-chunk-duration", type=float, default=30.0)
        parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
        hub_arguments(parser)
        args = parser.parse_args(arguments[1:])
        model = VADModel.load(args.checkpoint, resolve_device(args.device), **hub_options(args))
        print_result(
            model.segment(args.audio, args.output_dir, max_chunk_duration=args.max_chunk_duration)
        )
        return
    if arguments and arguments[0] == "detect":
        parser = argparse.ArgumentParser(prog="advai-vad detect")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--audio", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
        hub_arguments(parser)
        args = parser.parse_args(arguments[1:])
        result = VADModel.load(
            args.checkpoint, resolve_device(args.device), **hub_options(args)
        ).detect(args.audio)
        write_json(args.output, result)
        print_result({k: v for k, v in result.items() if k != "probabilities"})
        return
    return execute(
        arguments,
        kind="vad",
        load_config=load_config,
        model_class=VADModel,
        load_data=load_data,
        prepare_data=prepare_data,
        run=run,
        evaluate=evaluate,
        publish=publish,
        inference="detect-batch",
        infer_one=lambda model, row: model.detect(row["audio"] if isinstance(row, dict) else row),
    )
