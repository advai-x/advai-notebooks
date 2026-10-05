"""Hash task sources and their statically imported local dependencies without importing them."""

import ast
from importlib.util import resolve_name
from pathlib import Path

from .artifacts import digest


def source_fingerprint(kind, package_root=None):
    root = Path(package_root or Path(__file__).resolve().parents[1])
    pending = list((root / kind).rglob("*.py")) + [root / "__init__.py"]
    sources = {}

    def locate(module):
        if module == "advai_notebooks":
            return root / "__init__.py"
        if not module.startswith("advai_notebooks."):
            return None
        path = root.joinpath(*module.split(".")[1:])
        return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")

    while pending:
        path = pending.pop()
        if not path.is_file() or str(path.relative_to(root)) in sources:
            continue
        relative = path.relative_to(root)
        source = path.read_text(encoding="utf-8")
        sources[str(relative)] = source
        parts = list(relative.with_suffix("").parts)
        package = ".".join(["advai_notebooks", *parts[:-1]])
        for node in ast.walk(ast.parse(source)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = resolve_name("." * node.level + module, package)
                modules = [module, *(module + "." + alias.name for alias in node.names)]
            for module in modules:
                dependency = locate(module)
                if dependency is not None and dependency.is_file():
                    pending.append(dependency)
                    pending.extend(
                        p / "__init__.py"
                        for p in dependency.parents
                        if p != root and root in p.parents
                    )
    return digest(sources)
