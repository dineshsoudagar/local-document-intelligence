"""Pydantic models for health-related API routes."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response payload for basic service health checks."""

    status: str
    documents_dir: str
