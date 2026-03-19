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
        Learning:
        Returns the folder where managed document files will be stored.

        Prod:
        Managed document storage directory.
        """
        return self.base_dir / "documents"

    @property
    def metadata_dir(self) -> Path:
        """
        Learning:
        Returns the folder for metadata files like SQLite later.

        Prod:
        Metadata storage directory.
        """
        return self.base_dir / "metadata"

    @property
    def qdrant_dir(self) -> Path:
        """
        Learning:
        Returns the folder where local Qdrant data will live.

        Prod:
        Local Qdrant storage directory.
        """
        return self.base_dir / "qdrant"

    def ensure_exists(self) -> None:
        """
        Learning:
        Creates all required storage folders if they do not already exist.

        Prod:
        Create required storage directories.
        """
        # Learning:
        # `parents=True` creates missing parent folders too.
        # `exist_ok=True` avoids an error if the folder already exists.
        #
        # Prod:
        # Ensure required directories exist.
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)