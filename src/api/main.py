"""Application factory and process lifecycle management for the FastAPI server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_documents import router as documents_router
from src.api.routes_health import router as health_router
from src.api.routes_query import router as query_router
from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths
from src.config.api_config import ApiConfig
from src.config.generator_config import GeneratorConfig
from src.generation.answer_service import GroundedAnswerService
from src.indexing.index_service import IndexService
from src.retrieval.qwen_models import LocalQwenGenerator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services at startup and clean them up at shutdown."""
    paths = AppPaths(base_dir=Path("storage"))
    paths.ensure_exists()

    registry = DocumentRegistry(paths.documents_db_path)
    registry.initialize()

    index_service = IndexService()
    generator_config = GeneratorConfig()
    generator = LocalQwenGenerator(generator_config.generator_model_path)
    answer_service = GroundedAnswerService(
        index=index_service.index,
        config=generator_config,
        generator=generator,
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
    """Create and configure the FastAPI application instance."""
    api_config = ApiConfig()
    app = FastAPI(
        title="Offline Document Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[api_config.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(query_router)
    return app


app = create_app()


@app.get("/")
def root() -> dict[str, str]:
    """Return a minimal API status payload."""
    return {"message": "Offline Document Intelligence API is running."}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Return a minimal liveness payload for process managers and launchers."""
    return {"status": "ok"}
