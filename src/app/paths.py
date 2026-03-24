"""Application storage path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """Centralize the storage locations used by the application."""

    base_dir: Path

    @property
    def documents_dir(self) -> Path:
        """Return the managed document storage directory."""
        return self.base_dir / "documents"

    @property
    def metadata_dir(self) -> Path:
        """Return the metadata storage directory."""
        return self.base_dir / "metadata"

    @property
    def qdrant_dir(self) -> Path:
        """Return the local Qdrant storage directory."""
        return self.base_dir / "qdrant"

    def ensure_exists(self) -> None:
        """Create all required storage directories."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)

    @property
    def documents_db_path(self) -> Path:
        """Return the SQLite database path for document metadata."""
        return self.metadata_dir / "documents.db"
