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
            hub_arguments(item)
            item.add_argument("--split", choices=("validation", "test"), default="test")
            item.add_argument("--output", required=True)
    return parser, sub


def inference_arguments(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    hub_arguments(parser)


def hub_arguments(parser):
    parser.add_argument("--revision", help="Hub branch, tag or commit SHA for model loading")
    parser.add_argument("--cache-dir", help="Hub snapshot cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached Hub files only")


def hub_options(args):
    return {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }


def publish_parser(subparsers):
    parser = subparsers.add_parser("publish", help="Publish an inference export to Hugging Face")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--repo-id", required=True, help="namespace/model-name")
    parser.add_argument("--public", action="store_true", help="Use a public repository")
    parser.add_argument("--revision", default="main", help="Existing Hub branch")
    parser.add_argument("--license", help="Model-card license identifier; omitted by default")
    parser.add_argument("--model-card", help="Use a custom Markdown model card")
    parser.add_argument("--commit-message")
    parser.add_argument("--dry-run", action="store_true", help="Offline package validation only")
    parser.add_argument("--output", help="Save the publication result or offline upload plan")


def publish_result(args, publish):
    result = publish(
        args.export_dir,
        args.repo_id,
        private=not args.public,
        revision=args.revision,
        license=args.license,
        model_card=args.model_card,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )
    if args.output:
        from .artifacts import write_json

        write_json(args.output, result)
    print_result(result)


def read_inputs(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("Inference input must be a nonempty JSON list")
    return values


def print_result(result):
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
