"""Independent VM command-line entry point."""

from advai_notebooks.common.native_cli import main as execute

from .config import load_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .model import VMModel
from .pipeline import run
from .publish import publish


def infer_one(model, row):
    if not isinstance(row, dict):
        raise ValueError("Inference row must be an object")
    return model.rollout(row["observation"], row["actions"])


def main(argv=None):
    return execute(
        argv,
        kind="vm",
        load_config=load_config,
        model_class=VMModel,
        load_data=load_data,
        prepare_data=prepare_data,
        run=run,
        evaluate=evaluate,
        publish=publish,
        inference="rollout",
        infer_one=infer_one,
    )
