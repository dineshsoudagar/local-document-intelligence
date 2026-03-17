from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import snapshot_download

from src.config.model_catalog import ArtifactEntry, ModelCatalog, ModelEntry


MODEL_CATALOG = ModelCatalog()


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


def select_assets(only: Iterable[str] | None) -> tuple[list[ModelEntry], list[ArtifactEntry]]:
    """Select configured assets by key."""
    if not only:
        return list(MODEL_CATALOG.hf_models()), list(MODEL_CATALOG.artifacts())

    only_keys = set(only)
    selected_models = [
        entry for entry in MODEL_CATALOG.hf_models() if entry.key in only_keys
    ]
    selected_artifacts = [
        entry for entry in MODEL_CATALOG.artifacts() if entry.key in only_keys
    ]
    return selected_models, selected_artifacts


def ensure_empty_dir(path: Path) -> None:
    """Remove and recreate a directory."""
    if path.exists():
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
    models_root = project_root / MODEL_CATALOG.models_root
    models_root.mkdir(parents=True, exist_ok=True)

    token = os.getenv(args.token_env)
    selected_models, selected_artifacts = select_assets(args.only)

    records: list[dict[str, Any]] = []

    for entry in selected_models:
        print(f"Downloading {entry.key}: {entry.repo_id}")
        record = download_hf_model(
            entry=entry,
            project_root=project_root,
            token=token,
            force=args.force,
        )
        records.append(record)
        print(f"Saved to: {record['target_dir']}")

    for entry in selected_artifacts:
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
        "local_paths": MODEL_CATALOG.local_paths(project_root),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
