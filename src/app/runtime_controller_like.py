"""Lightweight protocol shared by the managed and bootstrap runtime controllers."""

from __future__ import annotations

from typing import Protocol


class RuntimeControllerLike(Protocol):
    """Behavior required by setup and health endpoints across runtime modes."""

    @property
    def services(self) -> object | None:
        """Return initialized runtime services when available."""

    @property
    def last_error(self) -> str | None:
        """Return the last runtime initialization error."""

    def clear_error(self) -> None:
        """Drop any previously recorded runtime initialization error."""

    def is_ready(self) -> bool:
        """Return whether the heavyweight runtime is initialized."""

    def reload(self) -> bool:
        """Refresh runtime services from persisted state."""

    def close(self) -> None:
        """Release initialized services."""

    def diagnostics(self) -> dict[str, str | bool | None]:
        """Return runtime and parser warmup diagnostics."""
