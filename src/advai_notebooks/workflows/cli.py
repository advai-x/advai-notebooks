import argparse
import json

from .config import load_config
from .retrieval import compare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="advai-retrieval")
    parser.add_argument("command", choices=["compare"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--embedding-run", required=True)
    parser.add_argument("--ranker-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare(load_config(args.config), args.embedding_run, args.ranker_run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
