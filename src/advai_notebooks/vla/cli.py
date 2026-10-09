"""Independent VLA command-line entry point."""

from advai_notebooks.common.native_cli import main as execute

from .config import load_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .model import VLAModel
from .pipeline import run
from .publish import publish


def infer_one(model, row):
    if not isinstance(row, dict):
        raise ValueError("Inference row must be an object")
    return model.predict(row["image"], row["instruction"], row["state"])


def main(argv=None):
    return execute(
        argv,
        kind="vla",
        load_config=load_config,
        model_class=VLAModel,
        load_data=load_data,
        prepare_data=prepare_data,
        run=run,
        evaluate=evaluate,
        publish=publish,
        inference="predict",
        infer_one=infer_one,
    )
