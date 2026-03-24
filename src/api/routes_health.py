"""Health-related API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.app_state import get_app_paths_from_state
from src.api.models_health import HealthResponse
from src.app.paths import AppPaths


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    paths: AppPaths = Depends(get_app_paths_from_state),
) -> HealthResponse:
    """Return a lightweight liveness response."""
    return HealthResponse(
        status="ok",
        documents_dir=str(paths.documents_dir),
    )


@router.get("/health/details")
def health_details(
    paths: AppPaths = Depends(get_app_paths_from_state),
) -> dict[str, str]:
    """Return the configured storage locations for diagnostics."""
    return {
        "base_dir": str(paths.base_dir),
        "documents_dir": str(paths.documents_dir),
        "metadata_dir": str(paths.metadata_dir),
        "qdrant_dir": str(paths.qdrant_dir),
    }
