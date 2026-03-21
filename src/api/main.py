from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.retrieval.qwen_models import LocalQwenGenerator
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
    # Create and prepare storage folders
    paths = AppPaths(base_dir=Path("storage"))
    paths.ensure_exists()

    # Open the SQLite document registry
    registry = DocumentRegistry(paths.documents_db_path)
    registry.initialize()

    # Build shared backend services once for the whole app
    index_service = IndexService()

    # Keep one generator config instance for startup and runtime
    generator_config = GeneratorConfig()

    # Preload the text generator into memory during app startup
    generator = LocalQwenGenerator(generator_config.generator_model_path)

    # Reuse the preloaded generator for all query requests
    answer_service = GroundedAnswerService(
        index=index_service.index,
        config=generator_config,
        generator=generator,
    )

    # Store shared objects on app.state so routes can access them through dependency injection
    app.state.paths = paths
    app.state.document_registry = registry
    app.state.index_service = index_service
    app.state.answer_service = answer_service

    try:
        yield
    finally:
        # Cleanly close index resources when the app shuts down
        index_service.close()


def create_app() -> FastAPI:
    # Create the FastAPI app
    app = FastAPI(
        title="Offline Document Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Allow the React dev server to call this backend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API route groups
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