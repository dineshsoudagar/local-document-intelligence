from __future__ import annotations

# Learning:
# This module defines the storage folders used by the app in one place.
#
# Prod:
# Application storage path helpers.

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """
    Learning:
    Holds the main storage paths used by the application.

    Instead of writing folder names in many files, we define them once here.
    This keeps path handling clean and consistent.

    Prod:
    Centralized application storage paths.
    """

    base_dir: Path

    @property
    def documents_dir(self) -> Path:
        """
        Managed document storage directory.
        """
        return self.base_dir / "documents"

    @property
    def metadata_dir(self) -> Path:
        """
        Metadata storage directory.
        """
        return self.base_dir / "metadata"

    @property
    def qdrant_dir(self) -> Path:
        """
        Local Qdrant storage directory.
        """
        return self.base_dir / "qdrant"

    def ensure_exists(self) -> None:
        """
        Create required storage directories.
        """
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)

    @property
    def documents_db_path(self) -> Path:
        """Return the SQLite database path for document metadata."""
        return self.metadata_dir / "documents.db"