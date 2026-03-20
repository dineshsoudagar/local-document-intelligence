# FastAPI application entrypoint.

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from src.indexing.index_service import IndexService
from src.api.routes_documents import router as documents_router
from src.api.routes_health import router as health_router
from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths


def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline Document Intelligence API",
        version="0.1.0",
    )

    app.state.paths = AppPaths(base_dir=Path("storage"))

    @app.on_event("startup")
    def on_startup() -> None:
        app.state.paths.ensure_exists()

        registry = DocumentRegistry(app.state.paths.documents_db_path)
        registry.initialize()
        app.state.document_registry = registry

    app.include_router(health_router)
    app.include_router(documents_router)
    app.state.index_service = IndexService()

    return app


app = create_app()
