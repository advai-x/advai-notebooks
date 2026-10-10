"""STT validation, training, evaluation, export, publication and transcription."""

from advai_notebooks.common.native_cli import main as execute

from .config import load_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .model import STTModel
from .pipeline import run
from .publish import publish


def main(argv=None):
    return execute(
        argv,
        kind="stt",
        load_config=load_config,
        model_class=STTModel,
        load_data=load_data,
        prepare_data=prepare_data,
        run=run,
        evaluate=evaluate,
        publish=publish,
        inference="transcribe",
        infer_one=lambda model, row: {
            "text": model.transcribe(row["audio"] if isinstance(row, dict) else row)
        },
    )
