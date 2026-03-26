"""Download configured local models and artifacts into the project models directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import snapshot_download

from src.config.model_catalog import (
    ArtifactEntry,
    ModelCatalog,
    ModelEntry,
    default_pipeline_models,
)


MODEL_CATALOG = ModelCatalog()
PIPELINE_MODELS = default_pipeline_models()

HF_CONFIG_MARKERS = (
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)

HF_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.onnx",
    "*.gguf",
)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local model and artifact download."""
    cli = argparse.ArgumentParser(
        description=(
            "Download configured Hugging Face models and Docling artifacts into the "
            "project-local models directory."
        ),
    )
    cli.add_argument(
        "--project-root",
        default=".",
        help="Project root that contains the models directory.",
    )
    cli.add_argument(
        "--only",
        nargs="+",
        choices=list(MODEL_CATALOG.downloadable_keys()),
        help="Subset of configured assets to download.",
    )

    cli.add_argument(
        "--all",
        action="store_true",
        help="Download every registered asset instead of only the selected pipeline assets.",
    )
    cli.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing local target directory before downloading.",
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
    cli.add_argument(
        "--docling-progress",
        action="store_true",
        help="Show Docling download progress when downloading Docling artifacts.",
    )
    cli.add_argument(
        "--docling-with-easyocr",
        action="store_true",
        help="Also download EasyOCR assets into the Docling artifacts directory.",
    )
    cli.add_argument(
        "--docling-with-rapidocr",
        dest="docling_with_rapidocr",
        action="store_true",
        default=True,
        help="Download RapidOCR assets into the Docling artifacts directory.",
    )
    cli.add_argument(
        "--no-docling-with-rapidocr",
        dest="docling_with_rapidocr",
        action="store_false",
        help="Skip RapidOCR assets when downloading Docling artifacts.",
    )
    cli.add_argument(
        "--docling-with-code-formula",
        dest="docling_with_code_formula",
        action="store_true",
        default=True,
        help="Download Docling code and formula assets.",
    )
    cli.add_argument(
        "--no-docling-with-code-formula",
        dest="docling_with_code_formula",
        action="store_false",
        help="Skip Docling code and formula assets.",
    )
    cli.add_argument(
        "--docling-with-picture-classifier",
        dest="docling_with_picture_classifier",
        action="store_true",
        default=True,
        help="Download Docling picture classifier assets.",
    )
    cli.add_argument(
        "--no-docling-with-picture-classifier",
        dest="docling_with_picture_classifier",
        action="store_false",
        help="Skip Docling picture classifier assets.",
    )
    cli.add_argument(
        "--docling-with-layout",
        dest="docling_with_layout",
        action="store_true",
        default=True,
        help="Download Docling layout assets.",
    )
    cli.add_argument(
        "--no-docling-with-layout",
        dest="docling_with_layout",
        action="store_false",
        help="Skip Docling layout assets.",
    )
    cli.add_argument(
        "--docling-with-tableformer",
        dest="docling_with_tableformer",
        action="store_true",
        default=True,
        help="Download Docling table structure assets.",
    )
    cli.add_argument(
        "--no-docling-with-tableformer",
        dest="docling_with_tableformer",
        action="store_false",
        help="Skip Docling table structure assets.",
    )
    return cli.parse_args()


def select_assets(
    only: Iterable[str] | None,
    *,
    all_assets: bool,
) -> tuple[list[ModelEntry], list[ArtifactEntry], tuple[str, ...]]:
    """Select assets by explicit keys or by the configured pipeline defaults."""
    if only:
        selected_keys = tuple(dict.fromkeys(only))
    elif all_assets:
        selected_keys = MODEL_CATALOG.downloadable_keys()
    else:
        selected_keys = PIPELINE_MODELS.required_asset_keys()

    selected_key_set = set(selected_keys)
    selected_models = [
        entry for entry in MODEL_CATALOG.hf_models() if entry.key in selected_key_set
    ]
    selected_artifacts = [
        entry for entry in MODEL_CATALOG.artifacts() if entry.key in selected_key_set
    ]
    return selected_models, selected_artifacts, selected_keys


def _has_any_matching_file(root: Path, patterns: Iterable[str]) -> bool:
    """Return whether the directory tree contains a file matching any pattern."""
    if not root.is_dir():
        return False

    for pattern in patterns:
        if any(path.is_file() for path in root.rglob(pattern)):
            return True
    return False


def is_hf_model_downloaded(entry: ModelEntry, project_root: Path) -> bool:
    """Return whether a Hugging Face model appears to be fully available locally."""
    target_dir = entry.resolve_dir(
        project_root=project_root,
        models_root=MODEL_CATALOG.models_root,
    )
    if not target_dir.is_dir():
        return False

    has_config = any((target_dir / marker).is_file() for marker in HF_CONFIG_MARKERS)
    if not has_config:
        return False

    return _has_any_matching_file(target_dir, HF_WEIGHT_PATTERNS)


def is_artifact_downloaded(entry: ArtifactEntry, project_root: Path) -> bool:
    """Return whether a non-Hugging Face artifact bundle exists locally."""
    target_dir = entry.resolve_dir(
        project_root=project_root,
        models_root=MODEL_CATALOG.models_root,
    )
    if not target_dir.is_dir():
        return False

    return any(target_dir.rglob("*"))


def build_existing_record(
    entry: ModelEntry | ArtifactEntry,
    project_root: Path,
) -> dict[str, Any]:
    """Return a manifest record for an asset that is already available locally."""
    target_dir = entry.resolve_dir(
        project_root=project_root,
        models_root=MODEL_CATALOG.models_root,
    )

    if isinstance(entry, ModelEntry):
        return {
            "key": entry.key,
            "source_type": "huggingface",
            "repo_id": entry.repo_id,
            "revision": entry.revision,
            "target_dir": str(target_dir.resolve()),
            "snapshot_path": str(target_dir.resolve()),
        }

    return {
        "key": entry.key,
        "source_type": "docling",
        "target_dir": str(target_dir.resolve()),
        "options": {
            "with_layout": None,
            "with_tableformer": None,
            "with_code_formula": None,
            "with_picture_classifier": None,
            "with_rapidocr": None,
            "with_easyocr": None,
        },
    }


def partition_selected_assets_by_presence(
    project_root: Path,
    selected_models: Iterable[ModelEntry],
    selected_artifacts: Iterable[ArtifactEntry],
) -> tuple[list[dict[str, Any]], list[ModelEntry], list[ArtifactEntry]]:
    """Split selected assets into already-present records and missing downloads."""
    existing_records: list[dict[str, Any]] = []
    missing_models: list[ModelEntry] = []
    missing_artifacts: list[ArtifactEntry] = []

    for entry in selected_models:
        if is_hf_model_downloaded(entry, project_root):
            existing_records.append(build_existing_record(entry, project_root))
        else:
            missing_models.append(entry)

    for entry in selected_artifacts:
        if is_artifact_downloaded(entry, project_root):
            existing_records.append(build_existing_record(entry, project_root))
        else:
            missing_artifacts.append(entry)

    return existing_records, missing_models, missing_artifacts


def ensure_empty_dir(path: Path) -> None:
    """Remove and recreate a directory."""
    if path.exists():
        # `--force` is intentionally destructive for the target directory so a
        # rerun cannot leave stale files from a previous snapshot behind.
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_hf_model(
    entry: ModelEntry,
    project_root: Path,
    token: str | None,
    force: bool,
) -> dict[str, Any]:
    """Download a single Hugging Face snapshot into the configured local directory."""
    target_dir = entry.resolve_dir(
        project_root=project_root,
        models_root=MODEL_CATALOG.models_root,
    )

    if force:
        ensure_empty_dir(target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    # Download directly into the final project-local directory so consumers can
    # read the model files without any extra copy or extraction step.
    snapshot_path = snapshot_download(
        repo_id=entry.repo_id,
        revision=entry.revision,
        local_dir=str(target_dir),
        token=token,
    )

    return {
        "key": entry.key,
        "source_type": "huggingface",
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "target_dir": str(target_dir.resolve()),
        "snapshot_path": str(Path(snapshot_path).resolve()),
    }


def download_docling_artifacts(
    entry: ArtifactEntry,
    project_root: Path,
    force: bool,
    *,
    progress: bool,
    with_layout: bool,
    with_tableformer: bool,
    with_code_formula: bool,
    with_picture_classifier: bool,
    with_rapidocr: bool,
    with_easyocr: bool,
) -> dict[str, Any]:
    """Download Docling pipeline artifacts into the configured local directory."""
    target_dir = entry.resolve_dir(
        project_root=project_root,
        models_root=MODEL_CATALOG.models_root,
    )

    if force:
        ensure_empty_dir(target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from docling.utils.model_downloader import download_models
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Install docling in the active environment before "
            "downloading Docling artifacts."
        ) from exc

    # Keep optional Docling components explicit here instead of relying on
    # library defaults so offline asset contents remain reproducible across
    # environments and Docling version upgrades.
    download_models(
        output_dir=target_dir,
        force=force,
        progress=progress,
        with_layout=with_layout,
        with_tableformer=with_tableformer,
        with_code_formula=with_code_formula,
        with_picture_classifier=with_picture_classifier,
        with_rapidocr=with_rapidocr,
        with_easyocr=with_easyocr,
        with_smolvlm=False,
        with_granitedocling=False,
        with_granitedocling_mlx=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
    )

    return {
        "key": entry.key,
        "source_type": "docling",
        "target_dir": str(target_dir.resolve()),
        "options": {
            "with_layout": with_layout,
            "with_tableformer": with_tableformer,
            "with_code_formula": with_code_formula,
            "with_picture_classifier": with_picture_classifier,
            "with_rapidocr": with_rapidocr,
            "with_easyocr": with_easyocr,
        },
    }


def write_manifest(
    models_root: Path,
    manifest_name: str,
    records: list[dict[str, Any]],
) -> Path:
    """Write a manifest describing all locally downloaded assets."""
    manifest_path = models_root / manifest_name
    # The manifest is the machine-readable source of truth for what was
    # downloaded and where it was materialized on disk for this project.
    payload = {
        "models_root": str(models_root.resolve()),
        "models": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    """Download configured assets into the project-local models directory."""
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    # Ensure the shared models root exists before any per-asset download runs so
    # manifest writing and summary reporting always have a stable base path.
    models_root = project_root / MODEL_CATALOG.models_root
    models_root.mkdir(parents=True, exist_ok=True)

    token = os.getenv(args.token_env)
    selected_models, selected_artifacts, selected_keys = select_assets(
        args.only,
        all_assets=args.all,
    )

    records: list[dict[str, Any]] = []
    models_to_download = selected_models
    artifacts_to_download = selected_artifacts

    if not args.force:
        (
            existing_records,
            models_to_download,
            artifacts_to_download,
        ) = partition_selected_assets_by_presence(
            project_root,
            selected_models,
            selected_artifacts,
        )
        records.extend(existing_records)

        for record in existing_records:
            print(f"Skipping {record['key']}: already present at {record['target_dir']}")

    for entry in models_to_download:
        print(f"Downloading {entry.key}: {entry.repo_id}")
        record = download_hf_model(
            entry=entry,
            project_root=project_root,
            token=token,
            force=args.force,
        )
        records.append(record)
        print(f"Saved to: {record['target_dir']}")

    for entry in artifacts_to_download:
        print(f"Downloading {entry.key}: Docling offline artifacts")
        record = download_docling_artifacts(
            entry=entry,
            project_root=project_root,
            force=args.force,
            progress=args.docling_progress,
            with_layout=args.docling_with_layout,
            with_tableformer=args.docling_with_tableformer,
            with_code_formula=args.docling_with_code_formula,
            with_picture_classifier=args.docling_with_picture_classifier,
            with_rapidocr=args.docling_with_rapidocr,
            with_easyocr=args.docling_with_easyocr,
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
        "selected_keys": selected_keys,
        "local_paths": MODEL_CATALOG.local_paths(project_root, selected_keys),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
