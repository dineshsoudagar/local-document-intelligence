"""Setup API routes for managed runtime provisioning."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.app_state import get_setup_service_from_state
from src.api.models_setup import (
    SetupOptionsResponse,
    SetupStartRequest,
    SetupStatusResponse,
)
from src.app.setup_service import SetupService


router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/status", response_model=SetupStatusResponse)
def get_setup_status(
    setup_service: SetupService = Depends(get_setup_service_from_state),
) -> SetupStatusResponse:
    """Return the persisted managed runtime setup status."""
    return SetupStatusResponse.model_validate(setup_service.get_status().model_dump())


@router.get("/options", response_model=SetupOptionsResponse)
def get_setup_options(
    setup_service: SetupService = Depends(get_setup_service_from_state),
) -> SetupOptionsResponse:
    """Return curated setup choices and compute recommendations."""
    return SetupOptionsResponse.model_validate(setup_service.get_options())


@router.post("/start", response_model=SetupStatusResponse)
def start_setup(
    request: SetupStartRequest,
    setup_service: SetupService = Depends(get_setup_service_from_state),
) -> SetupStatusResponse:
    """Start a new provisioning run for the managed runtime."""
    try:
        status_payload = setup_service.start_install(
            generator_key=request.generator_key,
            embedding_key=request.embedding_key,
            generator_load_preset=request.generator_load_preset,
            torch_variant=request.torch_variant,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return SetupStatusResponse.model_validate(status_payload.model_dump())


@router.post("/retry", response_model=SetupStatusResponse)
def retry_setup(
    setup_service: SetupService = Depends(get_setup_service_from_state),
) -> SetupStatusResponse:
    """Retry provisioning using the last persisted setup choices."""
    try:
        status_payload = setup_service.retry_install()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return SetupStatusResponse.model_validate(status_payload.model_dump())


@router.post("/cancel", response_model=SetupStatusResponse)
def cancel_setup(
    setup_service: SetupService = Depends(get_setup_service_from_state),
) -> SetupStatusResponse:
    """Request cancellation for an active provisioning run."""
    return SetupStatusResponse.model_validate(
        setup_service.cancel_install().model_dump()
    )
