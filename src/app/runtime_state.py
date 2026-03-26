"""Persisted runtime configuration and setup status models."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


RuntimeInstallState = Literal["not_ready", "installing", "ready", "failed"]
RUNTIME_VERSION = "1"


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ManagedAppConfig(BaseModel):
    """Persisted runtime choices used to initialize the application stack."""

    runtime_version: str = RUNTIME_VERSION
    install_state: RuntimeInstallState = "not_ready"
    selected_generator_key: str | None = None
    selected_embedding_key: str | None = None
    selected_generator_load_preset: str | None = None
    selected_torch_variant: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)

    def mark_updated(self) -> "ManagedAppConfig":
        """Return a copied config with an updated timestamp."""
        return self.model_copy(update={"updated_at": utc_now_iso()})


class SetupStatus(BaseModel):
    """Persisted installer status shown by the setup UI."""

    install_state: RuntimeInstallState = "not_ready"
    current_step: str | None = None
    progress_message: str | None = None
    last_error: str | None = None
    cancel_requested: bool = False
    is_busy: bool = False
    selected_generator_key: str | None = None
    selected_embedding_key: str | None = None
    selected_generator_load_preset: str | None = None
    selected_torch_variant: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)

    def with_updates(self, **updates: object) -> "SetupStatus":
        """Return a copied status with an updated timestamp."""
        payload = dict(updates)
        payload["updated_at"] = utc_now_iso()
        return self.model_copy(update=payload)


def load_managed_app_config(path: Path) -> ManagedAppConfig:
    """Load the persisted runtime config if it exists."""
    if not path.is_file():
        return ManagedAppConfig()
    return ManagedAppConfig.model_validate_json(path.read_text(encoding="utf-8"))


def save_managed_app_config(path: Path, config: ManagedAppConfig) -> ManagedAppConfig:
    """Persist the managed runtime config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = config.mark_updated()
    path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    return updated


def load_setup_status(path: Path) -> SetupStatus:
    """Load the persisted setup status if it exists."""
    if not path.is_file():
        return SetupStatus()
    return SetupStatus.model_validate_json(path.read_text(encoding="utf-8"))


def save_setup_status(path: Path, status: SetupStatus) -> SetupStatus:
    """Persist the setup status."""
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = status.with_updates()
    path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    return updated
