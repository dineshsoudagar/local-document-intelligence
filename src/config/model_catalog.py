from __future__ import annotations

"""Model catalog for local model locations used by the project."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Describe one model and where it should live locally."""
    key: str
    repo_id: str
    relative_dir: str
    revision: str = "main"

    def resolve_dir(
        self,
        project_root: str | Path,
        models_root: str,
    ) -> Path:
        """Resolve the local directory for this model entry."""
        return Path(project_root).resolve() / models_root / self.relative_dir


@dataclass(slots=True)
class ModelCatalog:
    """Catalog of local paths for embedder, reranker, and generator models."""
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

    def all(self) -> tuple[ModelEntry, ...]:
        """Return every registered model entry."""
        return (
            self.embedder,
            self.reranker,
            self.generator,
        )

    def get(self, key: str) -> ModelEntry:
        """Return one model entry by key."""
        for entry in self.all():
            if entry.key == key:
                return entry
        raise KeyError(f"Unknown model key: {key}")

    def local_paths(self, project_root: str | Path) -> dict[str, str]:
        """Resolve all model keys to concrete local directories."""
        root = Path(project_root).resolve()
        return {
            entry.key: str(entry.resolve_dir(root, self.models_root))
            for entry in self.all()
        }
