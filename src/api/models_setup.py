"""Pydantic models for managed runtime setup endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class SetupStartRequest(BaseModel):
    """Request payload for starting the managed runtime install."""

    generator_key: str
    embedding_key: str
    generator_load_preset: str
    torch_variant: str

    @field_validator(
        "generator_key",
        "embedding_key",
        "generator_load_preset",
        "torch_variant",
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Normalize and validate required setup selections."""
        value = value.strip()
        if not value:
            raise ValueError("setup selection values must not be empty.")
        return value


class SetupStatusResponse(BaseModel):
    """Response payload describing current setup progress."""

    install_state: str
    current_step: str | None
    progress_message: str | None
    last_error: str | None
    cancel_requested: bool
    is_busy: bool
    selected_generator_key: str | None
    selected_embedding_key: str | None
    selected_generator_load_preset: str | None
    selected_torch_variant: str | None
    started_at: str | None
    completed_at: str | None
    updated_at: str


class SetupOptionsResponse(BaseModel):
    """Response payload containing setup choices for the frontend."""

    generator_models: list[dict[str, Any]]
    embedding_models: list[dict[str, Any]]
    generator_load_presets: list[dict[str, Any]]
    compute: dict[str, Any]
    torch_variants: list[dict[str, Any]]
