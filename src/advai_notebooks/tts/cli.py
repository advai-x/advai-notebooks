"""TTS validation, training, evaluation, export, publication and waveform synthesis."""

from advai_notebooks.common.native_cli import main as execute

from .config import load_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .model import TTSModel
from .pipeline import run
from .publish import publish


def main(argv=None):
    return execute(
        argv,
        kind="tts",
        load_config=load_config,
        model_class=TTSModel,
        load_data=load_data,
        prepare_data=prepare_data,
        run=run,
        evaluate=evaluate,
        publish=publish,
        inference="synthesize",
        infer_one=lambda model, row: model.synthesize(row["text"], row["output_audio"]),
    )
