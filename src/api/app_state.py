"""Typed accessors for shared FastAPI application state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths
from src.app.runtime_controller import RuntimeController
from src.app.setup_service import SetupService

if TYPE_CHECKING:
    from src.generation.answer_service import GroundedAnswerService
    from src.indexing.index_service import IndexService


def _get_runtime_controller(request: Request) -> RuntimeController:
    """Return the shared runtime controller."""
    return request.app.state.runtime_controller


def _require_runtime_services(request: Request):
    """Return live runtime services or raise a setup-required error."""
    controller = _get_runtime_controller(request)
    services = controller.services
    if services is None:
        detail = controller.last_error or "setup_required: runtime is not ready yet."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )
    return services


def get_index_service_from_state(request: Request) -> "IndexService":
    """Return the shared index service stored on the FastAPI application."""
    return _require_runtime_services(request).index_service


def get_app_paths_from_state(request: Request) -> AppPaths:
    """Return the shared application path registry."""
    return request.app.state.paths


def get_document_registry_from_state(request: Request) -> DocumentRegistry:
    """Return the shared document registry."""
    return _require_runtime_services(request).document_registry


def get_answer_service_from_state(request: Request) -> "GroundedAnswerService":
    """Return the shared grounded answer service."""
    return _require_runtime_services(request).answer_service


def get_setup_service_from_state(request: Request) -> SetupService:
    """Return the managed runtime setup service."""
    return request.app.state.setup_service
