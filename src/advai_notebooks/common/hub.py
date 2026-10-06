"""Task-neutral Hub transport for explicitly selected inference artifacts."""

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import yaml

from .artifacts import write_json

MODEL_FILES = {
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "sentencepiece.model",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
}


def resolve_export(path, *, revision=None, token=None, cache_dir=None, local_files_only=False):
    """Resolve a local export or download a versioned Hub model snapshot."""
    local = Path(path).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.exists() or local.is_absolute() or str(path).startswith((".", "~")):
        raise FileNotFoundError(f"Model export directory does not exist: {path}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(path),
            repo_type="model",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )


def collect_files(root, *, names=(), directories=("",)):
    """Select model assets, excluding raw data, logs and training state."""
    root = Path(root).resolve()
    files = []
    for directory in sorted(set(directories)):
        relative = PurePosixPath(directory)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid export component directory: {directory}")
        folder = root / directory
        if not folder.resolve().is_relative_to(root) or folder.is_symlink():
            raise ValueError(f"Export component escapes its directory: {directory}")
        if not folder.is_dir():
            continue
        for file in sorted(folder.iterdir()):
            is_shard = (
                file.name.startswith("model-")
                and file.name.endswith(".safetensors")
                or file.name.startswith("pytorch_model-")
                and file.name.endswith(".bin")
            )
            if file.name not in MODEL_FILES | set(names) and not is_shard:
                continue
            if file.is_symlink():
                raise ValueError(f"Export assets must not be symbolic links: {file.name}")
            if file.is_file():
                files.append(file.relative_to(root).as_posix())
    return sorted(set(files))


def card(repo_id, task, body, *, library, license=None, tags=(), pipeline_tag=None):
    metadata = {"library_name": library, "tags": ["advai-notebooks", task, *tags]}
    if license:
        metadata["license"] = license
    if pipeline_tag:
        metadata["pipeline_tag"] = pipeline_tag
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False) + f"---\n\n# {repo_id}\n\n{body}\n"


def _metadata(value):
    if isinstance(value, dict):
        return {
            key: (
                "local"
                if key in {"id", "model_id", "_name_or_path", "name_or_path"}
                and isinstance(item, str)
                and Path(item).expanduser().is_absolute()
                else _metadata(item)
            )
            for key, item in value.items()
            if key not in {"token", "use_auth_token"}
        }
    if isinstance(value, list):
        return [_metadata(item) for item in value]
    return value


def publish_files(
    export_dir,
    repo_id,
    task,
    files,
    model_card,
    *,
    private=True,
    revision="main",
    token=None,
    dry_run=False,
    commit_message=None,
    validate_package=None,
):
    """Stage an inference package; dry runs never instantiate a network client."""
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    if repo_id.count("/") != 1:
        raise ValueError("repo_id must be namespace/model-name")
    if not revision or type(private) is not bool:
        raise ValueError("revision must be nonempty and private must be boolean")
    export_dir = Path(export_dir).resolve()
    with TemporaryDirectory(prefix="advai-publish-") as directory:
        staging = Path(directory)
        for name in files:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid inference artifact path: {name}")
            source, destination = export_dir / name, staging / name
            if source.is_symlink() or not source.resolve().is_relative_to(export_dir):
                raise ValueError(f"Invalid inference artifact: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.name in {
                "advai_metadata.json",
                "source.json",
                "config.json",
                "tokenizer_config.json",
                "config_sentence_transformers.json",
            }:
                write_json(destination, _metadata(json.loads(source.read_text())))
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
        (staging / "README.md").write_text(model_card, encoding="utf-8")
        manifest = []
        for file in sorted(p for p in staging.rglob("*") if p.is_file()):
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest.append(
                {
                    "path": file.relative_to(staging).as_posix(),
                    "bytes": file.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        package = {"schema_version": 1, "task": task, "files": manifest}
        write_json(staging / "publish_manifest.json", package)
        if validate_package is not None:
            validate_package(staging)
        result = {
            "status": "dry_run" if dry_run else "published",
            "task": task,
            "repo_id": repo_id,
            "private": private,
            "revision": revision,
            "files": manifest,
            "manifest_file": "publish_manifest.json",
            "model_card": model_card,
        }
        if dry_run:
            return result
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        if api.repo_info(repo_id=repo_id, repo_type="model").private != private:
            raise ValueError(
                "Existing Hub repository visibility differs from private/public setting"
            )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=staging,
            revision=revision,
            allow_patterns=[*[row["path"] for row in manifest], "publish_manifest.json"],
            commit_message=commit_message or f"Publish {task} inference export",
        )
        result.update(
            {
                "commit": commit.oid,
                "commit_url": commit.commit_url,
                "repo_url": f"https://huggingface.co/{repo_id}",
            }
        )
        return result
