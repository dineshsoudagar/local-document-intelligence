from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.api.routes_documents import router as documents_router
from src.api.routes_health import router as health_router
from src.api.routes_query import router as query_router
from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths
from src.config.generator_config import GeneratorConfig
from src.generation.answer_service import GroundedAnswerService
from src.indexing.index_service import IndexService


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths = AppPaths(base_dir=Path("storage"))
    paths.ensure_exists()

    registry = DocumentRegistry(paths.documents_db_path)
    registry.initialize()

    index_service = IndexService()
    answer_service = GroundedAnswerService(
        index=index_service.index,
        config=GeneratorConfig(),
    )

    app.state.paths = paths
    app.state.document_registry = registry
    app.state.index_service = index_service
    app.state.answer_service = answer_service

    try:
        yield
    finally:
        index_service.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline Document Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(query_router)
    return app


app = create_app()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Offline Document Intelligence API is running."}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
