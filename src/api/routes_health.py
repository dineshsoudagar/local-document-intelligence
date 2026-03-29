"""Health-related API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.app_state import (
    get_app_paths_from_state,
    get_runtime_controller_from_state,
    get_setup_service_from_state,
)
from src.api.models_health import HealthResponse
from src.app.paths import AppPaths
from src.app.runtime_controller_like import RuntimeControllerLike
from src.app.runtime_state import SetupStatus
from src.app.setup_service import SetupService


router = APIRouter()


def _build_health_details_payload(
    *,
    paths: AppPaths,
    setup_status: SetupStatus,
    runtime_controller: RuntimeControllerLike,
    runtime_mode: str,
    launcher_log_path: str,
    backend_log_path: str,
) -> dict[str, str | bool | None]:
    """Build one diagnostic payload describing runtime and logging state."""
    diagnostics = runtime_controller.diagnostics()
    return {
        "app_root": str(paths.app_root),
        "code_root": str(paths.code_root),
        "base_dir": str(paths.base_dir),
        "documents_dir": str(paths.documents_dir),
        "metadata_dir": str(paths.metadata_dir),
        "qdrant_dir": str(paths.qdrant_dir),
        "launcher_log_path": launcher_log_path,
        "backend_log_path": backend_log_path,
        "install_state": setup_status.install_state,
        "runtime_mode": runtime_mode,
        "runtime_initialized": diagnostics["runtime_initialized"],
        "runtime_last_error": diagnostics["runtime_last_error"],
        "parser_warmup_ran_in_process": diagnostics["parser_warmup_ran_in_process"],
        "parser_warmup_started_at": diagnostics["parser_warmup_started_at"],
        "parser_warmup_completed_at": diagnostics["parser_warmup_completed_at"],
        "parser_warmup_completed": diagnostics["parser_warmup_completed"],
        "parser_warmup_error": diagnostics["parser_warmup_error"],
    }


def _is_backend_ready(payload: dict[str, str | bool | None]) -> bool:
    """Return whether the packaged backend is fully ready for the desktop shell."""
    return bool(
        payload["install_state"] == "ready"
        and payload["runtime_initialized"]
        and payload["parser_warmup_completed"]
        and payload["app_root"]
        and payload["launcher_log_path"]
        and payload["backend_log_path"]
    )


@router.get("/health", response_model=HealthResponse)
def health(
    paths: AppPaths = Depends(get_app_paths_from_state),
    setup_service: SetupService = Depends(get_setup_service_from_state),
) -> HealthResponse:
    """Return a lightweight liveness response."""
    status_payload = setup_service.get_status()
    return HealthResponse(
        status="ok" if status_payload.install_state == "ready" else status_payload.install_state,
        documents_dir=str(paths.documents_dir),
    )


@router.get("/health/details")
def health_details(
    request: Request,
    paths: AppPaths = Depends(get_app_paths_from_state),
    setup_service: SetupService = Depends(get_setup_service_from_state),
    runtime_controller: RuntimeControllerLike = Depends(get_runtime_controller_from_state),
) -> dict[str, str | bool | None]:
    """Return the configured storage locations for diagnostics."""
    setup_status = setup_service.get_status()
    runtime_mode = str(getattr(request.app.state, "backend_runtime_mode", "unknown"))
    launcher_log_path = str(
        getattr(request.app.state, "launcher_log_path", paths.launcher_log_path)
    )
    backend_log_path = str(
        getattr(request.app.state, "backend_log_path", paths.backend_log_path)
    )
    return _build_health_details_payload(
        paths=paths,
        setup_status=setup_status,
        runtime_controller=runtime_controller,
        runtime_mode=runtime_mode,
        launcher_log_path=launcher_log_path,
        backend_log_path=backend_log_path,
    )


@router.get("/readyz")
def readyz(
    request: Request,
    paths: AppPaths = Depends(get_app_paths_from_state),
    setup_service: SetupService = Depends(get_setup_service_from_state),
    runtime_controller: RuntimeControllerLike = Depends(get_runtime_controller_from_state),
) -> JSONResponse:
    """Return whether the backend is fully ready for the packaged launcher."""
    payload = _build_health_details_payload(
        paths=paths,
        setup_status=setup_service.get_status(),
        runtime_controller=runtime_controller,
        runtime_mode=str(getattr(request.app.state, "backend_runtime_mode", "unknown")),
        launcher_log_path=str(
            getattr(request.app.state, "launcher_log_path", paths.launcher_log_path)
        ),
        backend_log_path=str(
            getattr(request.app.state, "backend_log_path", paths.backend_log_path)
        ),
    )
    is_ready = _is_backend_ready(payload)
    payload["status"] = "ok" if is_ready else "not_ready"
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content=payload,
    )
