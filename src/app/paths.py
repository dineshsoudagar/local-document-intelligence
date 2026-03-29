"""Application storage and runtime path helpers."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "LocalDocumentIntelligence"
APP_ROOT_ENV_VAR = "LDI_APP_ROOT"
CODE_ROOT_ENV_VAR = "LDI_CODE_ROOT"
BACKEND_RUNTIME_MODE_ENV_VAR = "LDI_BACKEND_RUNTIME_MODE"
LAUNCHER_LOG_PATH_ENV_VAR = "LDI_LAUNCHER_LOG_PATH"


def _can_prepare_directory(path: Path) -> bool:
    """Return whether the process can create and write inside one directory."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / ".ldi-write-test"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _default_app_root() -> Path:
    """Return the managed per-user application root."""
    configured = os.getenv(APP_ROOT_ENV_VAR)
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if _can_prepare_directory(configured_path):
            return configured_path

    candidates: list[Path] = []

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data).resolve() / APP_NAME)

    candidates.append(Path.home().resolve() / "AppData" / "Local" / APP_NAME)
    candidates.append(Path(tempfile.gettempdir()).resolve() / APP_NAME)

    for candidate in candidates:
        if _can_prepare_directory(candidate):
            return candidate

    return candidates[0]


def _default_code_root() -> Path:
    """Return the code payload root for source and frozen launches."""
    configured = os.getenv(CODE_ROOT_ENV_VAR)
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if getattr(sys, "frozen", False) and not (configured_path / "src").is_dir():
            internal_dir = configured_path / "_internal"
            if (internal_dir / "src").is_dir():
                return internal_dir
        return configured_path

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        executable_dir = Path(sys.executable).resolve().parent
        internal_dir = executable_dir / "_internal"
        if (internal_dir / "src").is_dir():
            return internal_dir
        return executable_dir

    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Centralize mutable runtime and bundled payload locations."""

    app_root: Path
    code_root: Path

    @classmethod
    def from_default_locations(cls) -> "AppPaths":
        """Build the path registry from the default runtime roots."""
        return cls(
            app_root=_default_app_root(),
            code_root=_default_code_root(),
        )

    @property
    def base_dir(self) -> Path:
        """Return the storage root used by document and index data."""
        return self.app_root / "storage"

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

    @property
    def temp_dir(self) -> Path:
        """Return the temporary working directory."""
        return self.base_dir / "tmp"

    @property
    def config_dir(self) -> Path:
        """Return the persisted config directory."""
        return self.app_root / "config"

    @property
    def logs_dir(self) -> Path:
        """Return the runtime log directory."""
        return self.app_root / "logs"

    @property
    def runtime_dir(self) -> Path:
        """Return the managed runtime directory."""
        return self.app_root / "runtime"

    @property
    def embedded_python_dir(self) -> Path:
        """Return the bundled Python runtime directory."""
        return self.runtime_dir / "python"

    @property
    def managed_venv_dir(self) -> Path:
        """Return the managed virtual environment directory."""
        return self.runtime_dir / "venv"

    @property
    def models_dir(self) -> Path:
        """Return the managed models root."""
        return self.app_root / "models"

    @property
    def setup_status_path(self) -> Path:
        """Return the persisted setup status JSON path."""
        return self.config_dir / "setup-status.json"

    @property
    def runtime_config_path(self) -> Path:
        """Return the persisted runtime config JSON path."""
        return self.config_dir / "runtime-config.json"

    @property
    def launcher_log_path(self) -> Path:
        """Return the desktop launcher log file."""
        return self.logs_dir / "launcher.log"

    @property
    def backend_log_path(self) -> Path:
        """Return the backend diagnostic log file."""
        return self.logs_dir / "backend.log"

    @property
    def documents_db_path(self) -> Path:
        """Return the SQLite database path for document metadata."""
        return self.metadata_dir / "documents.db"

    @property
    def frontend_dist_dir(self) -> Path:
        """Return the built frontend distribution directory."""
        source_dist_dir = self.code_root / "frontend" / "dist"
        if source_dist_dir.is_dir():
            return source_dist_dir
        return self.app_root / "frontend" / "dist"

    @property
    def frontend_index_path(self) -> Path:
        """Return the main built frontend HTML entrypoint."""
        return self.frontend_dist_dir / "index.html"

    @property
    def bundled_wheelhouse_dir(self) -> Path:
        """Return the bundled offline wheelhouse directory."""
        return self.code_root / "bundle" / "wheels"

    @property
    def bundled_python_dir(self) -> Path:
        """Return the bundled bootstrap Python directory."""
        return self.code_root / "bundle" / "python"

    @property
    def requirements_path(self) -> Path:
        """Return the runtime requirements file."""
        return self.code_root / "requirements.txt"

    @property
    def download_models_script_path(self) -> Path:
        """Return the model download script entrypoint."""
        return self.code_root / "scripts" / "download_models.py"

    def ensure_exists(self) -> None:
        """Create the directories needed by the managed runtime."""
        for path in (
            self.app_root,
            self.base_dir,
            self.documents_dir,
            self.metadata_dir,
            self.qdrant_dir,
            self.temp_dir,
            self.config_dir,
            self.logs_dir,
            self.runtime_dir,
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
