"""Whisper task operations injected into the neutral epoch lifecycle."""

from advai_notebooks.common.native_training import run as execute

from .config import validate_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .losses import loss
from .model import STTModel


def run(cfg, resume=None, *, callbacks=()):
    validate_config(cfg)
    return execute(
        cfg, STTModel, load_data, prepare_data, loss, evaluate, resume=resume, callbacks=callbacks
    )
