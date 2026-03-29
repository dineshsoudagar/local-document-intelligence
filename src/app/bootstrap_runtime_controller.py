"""Minimal runtime controller used by the packaged bootstrap backend."""

from __future__ import annotations

from src.app.paths import AppPaths
from src.app.runtime_state import load_managed_app_config


class BootstrapRuntimeController:
    """Expose setup/health diagnostics without importing the heavy runtime stack."""

    def __init__(self, paths: AppPaths) -> None:
        self._paths = paths
        self._last_error: str | None = None

    @property
    def services(self) -> None:
        """The bootstrap backend never hosts live query/index services."""
        return None

    @property
    def last_error(self) -> str | None:
        """Return the last bootstrap runtime error."""
        return self._last_error

    def clear_error(self) -> None:
        """Drop any previously recorded bootstrap runtime error."""
        self._last_error = None

    def is_ready(self) -> bool:
        """The bootstrap backend is never the fully initialized runtime."""
        return False

    def reload(self) -> bool:
        """Refresh persisted config state without initializing heavy services."""
        self.clear_error()
        return False

    def close(self) -> None:
        """Release bootstrap runtime resources."""
        return None

    def diagnostics(self) -> dict[str, str | bool | None]:
        """Return health diagnostics compatible with the full runtime controller."""
        config = load_managed_app_config(self._paths.runtime_config_path)
        return {
            "runtime_initialized": False,
            "runtime_last_error": self._last_error,
            "runtime_install_state": config.install_state,
            "parser_warmup_ran_in_process": False,
            "parser_warmup_started_at": None,
            "parser_warmup_completed_at": None,
            "parser_warmup_completed": False,
            "parser_warmup_error": None,
        }
