from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

from src.config.model_catalog import ModelCatalog, ModelEntry


MODEL_CATALOG = ModelCatalog()


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Download configured model snapshots into the local project models directory.",
    )
    cli.add_argument(
        "--project-root",
        default=".",
        help="Project root that contains the models directory.",
    )
    cli.add_argument(
        "--only",
        nargs="+",
        choices=[entry.key for entry in MODEL_CATALOG.all()],
        help="Subset of configured models to download.",
    )
    cli.add_argument(
        "--force",
        action="store_true",
        help="Delete the local target directory before downloading.",
    )
    cli.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable that stores a Hugging Face token if needed.",
    )
    cli.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest file written inside the models root.",
    )
    return cli.parse_args()


def select_models(only: Iterable[str] | None) -> list[ModelEntry]:
    if not only:
        return list(MODEL_CATALOG.all())

    only_keys = set(only)
    return [entry for entry in MODEL_CATALOG.all() if entry.key in only_keys]


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_model(
    entry: ModelEntry,
    project_root: Path,
    token: str | None,
    force: bool,
) -> dict[str, str]:
    target_dir = entry.resolve_dir(
        project_root=project_root,
        models_root=MODEL_CATALOG.models_root,
    )

    if force:
        ensure_empty_dir(target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=entry.repo_id,
        revision=entry.revision,
        local_dir=str(target_dir),
        token=token,
    )

    return {
        "key": entry.key,
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "target_dir": str(target_dir.resolve()),
        "snapshot_path": str(Path(snapshot_path).resolve()),
    }


def write_manifest(
    models_root: Path,
    manifest_name: str,
    records: list[dict[str, str]],
) -> Path:
    manifest_path = models_root / manifest_name
    payload = {
        "models_root": str(models_root.resolve()),
        "models": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    models_root = project_root / MODEL_CATALOG.models_root
    models_root.mkdir(parents=True, exist_ok=True)

    token = os.getenv(args.token_env)
    selected_models = select_models(args.only)

    records: list[dict[str, str]] = []
    for entry in selected_models:
        print(f"Downloading {entry.key}: {entry.repo_id}")
        record = download_model(
            entry=entry,
            project_root=project_root,
            token=token,
            force=args.force,
        )
        records.append(record)
        print(f"Saved to: {record['target_dir']}")

    manifest_path = write_manifest(
        models_root=models_root,
        manifest_name=args.manifest_name,
        records=records,
    )

    summary = {
        "project_root": str(project_root),
        "models_root": str(models_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "local_paths": MODEL_CATALOG.local_paths(project_root),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
