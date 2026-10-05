"""Argument and JSON utilities with no task imports."""

import argparse
import json
from pathlib import Path


def experiment_parser(prog):
    parser = argparse.ArgumentParser(prog=prog)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "evaluate"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--root")
        if name == "run":
            item.add_argument("--resume", help="This experiment's output_dir/latest")
        if name == "evaluate":
            item.add_argument("--checkpoint", required=True)
            item.add_argument("--split", choices=("validation", "test"), default="test")
            item.add_argument("--output", required=True)
    return parser, sub


def inference_arguments(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")


def read_inputs(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("Inference input must be a nonempty JSON list")
    return values


def print_result(result):
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
