"""Compatibility loader; new callers use task-local configuration modules."""


def load_config(path, kind, root=None):
    if kind == "embedding":
        from advai_notebooks.embedding.config import load_config as load
    elif kind == "ranker":
        from advai_notebooks.ranker.config import load_config as load
    else:
        raise ValueError("kind must be embedding or ranker")
    return load(path, root=root)
