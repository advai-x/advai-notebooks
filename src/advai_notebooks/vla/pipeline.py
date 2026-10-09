"""Independent VLA lifecycle; no text-only LLM trainer imports."""

from advai_notebooks.common.native_training import run as execute
from advai_notebooks.common.task_configuration import validate_native

from .config import validate_config
from .data import load_data, prepare_data
from .evaluate import evaluate
from .losses import loss
from .model import VLAModel


def run(cfg, resume=None, *, callbacks=()):
    validate_native(cfg)
    validate_config(cfg)
    return execute(
        cfg, VLAModel, load_data, prepare_data, loss, evaluate, resume=resume, callbacks=callbacks
    )
