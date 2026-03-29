"""FastAPI application entrypoint and shared service initialization."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.api.routes_documents import router as documents_router
from src.api.routes_health import router as health_router
from src.api.routes_query import router as query_router
from src.api.routes_setup import router as setup_router
from src.app.backend_logging import configure_backend_logging
from src.app.paths import (
    AppPaths,
    BACKEND_RUNTIME_MODE_ENV_VAR,
    LAUNCHER_LOG_PATH_ENV_VAR,
)
from src.app.runtime_controller import RuntimeController
from src.app.setup_service import SetupService
from src.config.api_config import ApiConfig

API_PREFIXES = {
    "documents",
    "query",
    "setup",
    "health",
    "healthz",
    "readyz",
    "docs",
    "redoc",
    "openapi.json",
}


def _frontend_is_built(paths: AppPaths) -> bool:
    """Return whether the packaged frontend exists."""
    return paths.frontend_index_path.is_file()


def _resolve_frontend_file(paths: AppPaths, request_path: str) -> Path | None:
    """Resolve a safe frontend asset path under the built distribution directory."""
    if not request_path:
        return paths.frontend_index_path if _frontend_is_built(paths) else None

    dist_dir = paths.frontend_dist_dir.resolve()
    candidate = (dist_dir / request_path).resolve()

    if candidate != dist_dir and dist_dir not in candidate.parents:
        return None

    return candidate if candidate.is_file() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services at startup and clean them up at shutdown."""
    paths = AppPaths.from_default_locations()
    paths.ensure_exists()
    backend_log_path = configure_backend_logging(paths)

    backend_runtime_mode = os.getenv(BACKEND_RUNTIME_MODE_ENV_VAR, "unknown")
    launcher_log_path = os.getenv(
        LAUNCHER_LOG_PATH_ENV_VAR,
        str(paths.launcher_log_path),
    )

    logging.getLogger(__name__).info(
        "Backend lifespan startup app_root=%s code_root=%s runtime_mode=%s "
        "launcher_log_path=%s backend_log_path=%s",
        paths.app_root,
        paths.code_root,
        backend_runtime_mode,
        launcher_log_path,
        backend_log_path,
    )

    runtime_controller = RuntimeController(paths)
    setup_service = SetupService(paths, runtime_controller)
    runtime_controller.reload()

    app.state.paths = paths
    app.state.runtime_controller = runtime_controller
    app.state.setup_service = setup_service
    app.state.backend_runtime_mode = backend_runtime_mode
    app.state.launcher_log_path = launcher_log_path
    app.state.backend_log_path = str(backend_log_path)
    app.state.document_upload_count = 0

    try:
        yield
    finally:
        logging.getLogger(__name__).info("Backend lifespan shutdown.")
        runtime_controller.close()


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
    app.include_router(setup_router)
    app.include_router(documents_router)
    app.include_router(query_router)
    return app


app = create_app()


@app.get("/", include_in_schema=False)
def root():
    """Serve the packaged frontend or a basic API status payload."""
    paths = AppPaths.from_default_locations()
    if _frontend_is_built(paths):
        return FileResponse(paths.frontend_index_path)

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

    paths = AppPaths.from_default_locations()
    frontend_file = _resolve_frontend_file(paths, frontend_path)
    if frontend_file is not None:
        return FileResponse(frontend_file)

    if _frontend_is_built(paths):
        return FileResponse(paths.frontend_index_path)

    raise HTTPException(
        status_code=404,
        detail=(
            "Built frontend not found. Run 'npm run build' in the frontend directory "
            "before launching the packaged app."
        ),
    )
