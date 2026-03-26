"""FastAPI application entrypoint and shared service initialization."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.api.routes_documents import router as documents_router
from src.api.routes_health import router as health_router
from src.api.routes_query import router as query_router
from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths
from src.config.api_config import ApiConfig
from src.generation.answer_service import GroundedAnswerService
from src.indexing.index_service import IndexService
from src.retrieval.qwen_models import LocalQwenGenerator
from docling.datamodel.base_models import InputFormat

from src.config.generator_config import GeneratorConfig
from src.config.index_config import IndexConfig
from src.config.model_catalog import ModelCatalog, default_pipeline_models
from src.config.parser_config import ParserConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX_PATH = FRONTEND_DIST_DIR / "index.html"
API_PREFIXES = {"documents", "query", "health", "healthz", "docs", "redoc", "openapi.json"}


def _frontend_is_built() -> bool:
    """Return whether the packaged frontend exists."""
    return FRONTEND_INDEX_PATH.is_file()


def _resolve_frontend_file(request_path: str) -> Path | None:
    """Resolve a safe frontend asset path under the built distribution directory."""
    if not request_path:
        return FRONTEND_INDEX_PATH if _frontend_is_built() else None

    dist_dir = FRONTEND_DIST_DIR.resolve()
    candidate = (dist_dir / request_path).resolve()

    if candidate != dist_dir and dist_dir not in candidate.parents:
        return None

    return candidate if candidate.is_file() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services at startup and clean them up at shutdown."""
    paths = AppPaths(base_dir=Path("storage"))
    paths.ensure_exists()

    model_catalog = ModelCatalog()
    pipeline_models = default_pipeline_models()

    registry = DocumentRegistry(paths.documents_db_path)
    registry.initialize()

    index_config = IndexConfig(
        model_catalog=model_catalog,
        pipeline_models=pipeline_models,
    )
    parser_config = ParserConfig(
        model_catalog=model_catalog,
        pipeline_models=pipeline_models,
        allowed_formats=[InputFormat.PDF],
        enable_picture_description=False,
        include_picture_chunks=True,
    )
    index_service = IndexService(
        index_config=index_config,
        parser_config=parser_config,
    )

    generator_config = GeneratorConfig(
        model_catalog=model_catalog,
        pipeline_models=pipeline_models,
    ).with_load_preset("balanced_8bit")

    generator = LocalQwenGenerator.from_config(generator_config)
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


@app.get("/", include_in_schema=False)
def root():
    """Serve the packaged frontend or a basic API status payload."""
    if _frontend_is_built():
        return FileResponse(FRONTEND_INDEX_PATH)

    return JSONResponse({"message": "Offline Document Intelligence API is running."})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Return a lightweight process health response."""
    return {"status": "ok"}


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend(frontend_path: str):
    """Serve packaged frontend assets with SPA fallback behavior."""
    first_segment = frontend_path.split("/", 1)[0]
    if first_segment in API_PREFIXES:
        raise HTTPException(status_code=404, detail="Not found.")

    frontend_file = _resolve_frontend_file(frontend_path)
    if frontend_file is not None:
        return FileResponse(frontend_file)

    if _frontend_is_built():
        return FileResponse(FRONTEND_INDEX_PATH)

    raise HTTPException(
        status_code=404,
        detail=(
            "Built frontend not found. Run 'npm run build' in the frontend directory "
            "before launching the packaged app."
        ),
    )
