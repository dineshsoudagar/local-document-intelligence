"""Typed accessors for shared FastAPI application state."""

from __future__ import annotations

from fastapi import Request

from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths
from src.generation.answer_service import GroundedAnswerService
from src.indexing.index_service import IndexService


def get_index_service_from_state(request: Request) -> IndexService:
    """Return the shared index service stored on the FastAPI application."""
    return request.app.state.index_service


def get_app_paths_from_state(request: Request) -> AppPaths:
    """Return the shared application path registry."""
    return request.app.state.paths


def get_document_registry_from_state(request: Request) -> DocumentRegistry:
    """Return the shared document registry."""
    return request.app.state.document_registry


def get_answer_service_from_state(request: Request) -> GroundedAnswerService:
    """Return the shared grounded answer service."""
    return request.app.state.answer_service
