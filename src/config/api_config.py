"""Runtime settings for the FastAPI server."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ApiConfig:
    """Configuration for API runtime behavior such as CORS."""

    frontend_host: str = "localhost"
    frontend_port: int = 5173

    @property
    def frontend_origin(self) -> str:
        """Return the single allowed frontend dev origin."""
        return f"http://{self.frontend_host}:{self.frontend_port}"
