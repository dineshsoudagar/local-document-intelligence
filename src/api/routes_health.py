from __future__ import annotations

# Learning:
# This module contains health-related API routes.
#
# Prod:
# Health check routes.

from fastapi import APIRouter, Depends

from src.api.app_state import get_app_paths_from_state
from src.api.models_health import HealthResponse
from src.app.paths import AppPaths


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    paths: AppPaths = Depends(get_app_paths_from_state),
) -> HealthResponse:
    """
    Learning:
    This endpoint proves the server is running and that dependency injection works.

    Prod:
    Liveness endpoint for basic service health checks.
    """
    return HealthResponse(
        status="ok",
        documents_dir=str(paths.documents_dir),
    )


@router.get("/health/details")
def health_details(
    paths: AppPaths = Depends(get_app_paths_from_state),
) -> dict[str, str]:
    """
    Learning:
    This endpoint returns all storage paths so we can inspect app setup.

    Prod:
    Return basic application storage details.
    """
    return {
        "base_dir": str(paths.base_dir),
        "documents_dir": str(paths.documents_dir),
        "metadata_dir": str(paths.metadata_dir),
        "qdrant_dir": str(paths.qdrant_dir),
    }