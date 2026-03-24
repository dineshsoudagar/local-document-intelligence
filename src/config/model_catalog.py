"""Registry of local model and artifact locations used by the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Definition for a Hugging Face model stored under the project models directory."""

    key: str
    repo_id: str
    relative_dir: str
    revision: str = "main"

    def resolve_dir(self, project_root: str | Path, models_root: str) -> Path:
        """Return the resolved local directory for this model."""
        return Path(project_root).resolve() / models_root / self.relative_dir


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """Definition for non-Hugging Face assets stored under the project models directory."""

    key: str
    relative_dir: str

    def resolve_dir(self, project_root: str | Path, models_root: str) -> Path:
        """Return the resolved local directory for this artifact bundle."""
        return Path(project_root).resolve() / models_root / self.relative_dir


@dataclass(slots=True)
class ModelCatalog:
    """Central registry for all local model and artifact locations used by the app."""

    models_root: str = "models"

    embedder: ModelEntry = field(
        default_factory=lambda: ModelEntry(
            key="embedder",
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            relative_dir="embedders/qwen3-embedding-0.6b",
        )
    )
    reranker: ModelEntry = field(
        default_factory=lambda: ModelEntry(
            key="reranker",
            repo_id="Qwen/Qwen3-Reranker-0.6B",
            relative_dir="rerankers/qwen3-reranker-0.6b",
        )
    )
    generator: ModelEntry = field(
        default_factory=lambda: ModelEntry(
            key="generator",
            repo_id="Qwen/Qwen3-4B-Instruct-2507",
            relative_dir="generators/qwen3-4b-instruct-2507",
        )
    )
    picture_description: ModelEntry = field(
        default_factory=lambda: ModelEntry(
            key="picture_description",
            repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
            relative_dir="vlm/smolvlm-256m-instruct",
        )
    )
    docling_artifacts: ArtifactEntry = field(
        default_factory=lambda: ArtifactEntry(
            key="docling_artifacts",
            relative_dir="docling/artifacts",
        )
    )

    def hf_models(self) -> tuple[ModelEntry, ...]:
        """Return all Hugging Face-backed model entries."""
        return (
            self.embedder,
            self.reranker,
            self.generator,
            self.picture_description,
        )

    def artifacts(self) -> tuple[ArtifactEntry, ...]:
        """Return all non-Hugging Face artifact entries."""
        return (self.docling_artifacts,)

    def all(self) -> tuple[ModelEntry | ArtifactEntry, ...]:
        """Return all configured models and artifact bundles."""
        return self.hf_models() + self.artifacts()

    def downloadable_keys(self) -> tuple[str, ...]:
        """Return all keys supported by the downloader."""
        return tuple(entry.key for entry in self.all())

    def get(self, key: str) -> ModelEntry | ArtifactEntry:
        """Return a configured model or artifact entry by key."""
        for entry in self.all():
            if entry.key == key:
                return entry
        raise KeyError(f"Unknown model key: {key}")

    def get_hf_model(self, key: str) -> ModelEntry:
        """Return a Hugging Face model entry by key."""
        entry = self.get(key)
        if not isinstance(entry, ModelEntry):
            raise KeyError(f"Key '{key}' is not a Hugging Face model entry")
        return entry

    def get_artifact(self, key: str) -> ArtifactEntry:
        """Return a non-Hugging Face artifact entry by key."""
        entry = self.get(key)
        if not isinstance(entry, ArtifactEntry):
            raise KeyError(f"Key '{key}' is not an artifact entry")
        return entry

    def local_paths(self, project_root: str | Path) -> dict[str, str]:
        """Return resolved local paths for all configured assets."""
        root = Path(project_root).resolve()
        return {
            entry.key: str(entry.resolve_dir(root, self.models_root))
            for entry in self.all()
        }

    def embedder_path(self, project_root: str | Path) -> Path:
        """Return the resolved local embedder directory."""
        return self.embedder.resolve_dir(project_root, self.models_root)

    def reranker_path(self, project_root: str | Path) -> Path:
        """Return the resolved local reranker directory."""
        return self.reranker.resolve_dir(project_root, self.models_root)

    def generator_path(self, project_root: str | Path) -> Path:
        """Return the resolved local generator directory."""
        return self.generator.resolve_dir(project_root, self.models_root)

    def picture_description_path(self, project_root: str | Path) -> Path:
        """Return the resolved local picture-description model directory."""
        return self.picture_description.resolve_dir(project_root, self.models_root)

    def docling_artifacts_path(self, project_root: str | Path) -> Path:
        """Return the resolved local Docling artifacts directory."""
        return self.docling_artifacts.resolve_dir(project_root, self.models_root)

    def chunk_tokenizer_path(self, project_root: str | Path) -> Path:
        """Return the tokenizer path used for chunking."""
        return self.embedder_path(project_root)
