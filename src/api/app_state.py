# Helpers for reading shared application state.

from __future__ import annotations

from fastapi import Request
from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths
from src.indexing.index_service import IndexService


def get_index_service_from_state(request: Request) -> IndexService:
    return request.app.state.index_service


def get_app_paths_from_state(request: Request) -> AppPaths:
    return request.app.state.paths


def get_document_registry_from_state(request: Request) -> DocumentRegistry:
    return request.app.state.document_registry
