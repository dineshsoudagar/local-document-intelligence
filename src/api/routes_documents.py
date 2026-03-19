from __future__ import annotations

# Learning:
# This module contains document-related API routes.
#
# Prod:
# Document routes.

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from src.api.models_documents import (
    DocumentIngestRequest,
    DocumentIngestResponse,
)

router = APIRouter()


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
def ingest_document(
    request: DocumentIngestRequest,
) -> DocumentIngestResponse:
    """
    Learning:
    The request model handles only basic input validation.
    File-system checks happen here as runtime validation.

    Prod:
    Accept a document ingest request.
    """
    # Learning:
    # Convert the incoming string into a Path object for file checks.
    #
    # Prod:
    # Resolve the requested source path.
    source_path = Path(request.source_path)

    if source_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="source_path must point to a PDF file.",
        )

    if not source_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="source_path does not exist.",
        )

    if not source_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="source_path must point to a file.",
        )

    return DocumentIngestResponse(
        message="Received ingest request.",
        source_path=str(source_path),
    )