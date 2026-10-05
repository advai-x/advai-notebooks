"""Data preparation is an explicit command, separate from either trainer."""

import argparse
import json

from .config import load_config
from .pipeline import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="advai-data")
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(load_config(args.config, args.root)), ensure_ascii=False, indent=2))
