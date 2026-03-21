from __future__ import annotations

import shutil
from pathlib import Path
import hashlib
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
from src.api.app_state import (
    get_app_paths_from_state,
    get_document_registry_from_state,
    get_index_service_from_state,
)
from src.indexing.index_service import IndexService
from src.api.document_models import (
    DocumentDeleteResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentsListResponse,
    DocumentSummaryResponse,
    DocumentReindexResponse
)
from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths

router = APIRouter()


def _compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_doc_id(file_hash: str) -> str:
    return f"doc_{file_hash[:16]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_managed_document_path(paths: AppPaths, doc_id: str, suffix: str) -> Path:
    return paths.documents_dir / f"{doc_id}{suffix.lower()}"


# `@router.get("/documents", response_model=DocumentsListResponse)` registers
# an HTTP GET endpoint at /documents.
# `response_model=...` tells FastAPI what JSON shape this route should return,
# and FastAPI uses that for validation and OpenAPI docs.
@router.get("/documents", response_model=DocumentsListResponse)
def list_documents(
        # FastAPI dependency injection:
        # `Depends(get_document_registry_from_state)` tells FastAPI to call that function
        # and give the returned DocumentRegistry object to this route.
        # This keeps route code clean and avoids manually reading request.app.state here.
        registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> DocumentsListResponse:
    return DocumentsListResponse(
        items=[
            DocumentSummaryResponse.from_record(record)
            for record in registry.list_documents()
        ]
    )


# A route can receive more than one dependency.
# Here FastAPI injects both:
# - the shared DocumentRegistry
# - the shared AppPaths object
# That lets the route use registry logic and storage paths without creating either one itself.
@router.post("/documents/ingest", response_model=DocumentIngestResponse)
def ingest_document(
        request: DocumentIngestRequest,
        registry: DocumentRegistry = Depends(get_document_registry_from_state),
        paths: AppPaths = Depends(get_app_paths_from_state),
        index_service: IndexService = Depends(get_index_service_from_state),
) -> DocumentIngestResponse:
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

    source_path = source_path.resolve()

    file_hash = _compute_file_hash(source_path)
    doc_id = _build_doc_id(file_hash)

    existing = registry.get_document(doc_id)
    if existing is not None:
        return DocumentIngestResponse(
            message="Document already registered.",
            deduplicated=True,
            document=DocumentSummaryResponse.from_record(existing),
        )

    managed_path = _build_managed_document_path(
        paths=paths,
        doc_id=doc_id,
        suffix=source_path.suffix,
    )

    paths.documents_dir.mkdir(parents=True, exist_ok=True)

    if managed_path.resolve() != source_path:
        shutil.copy2(source_path, managed_path)

    ingested_at = _utc_now_iso()

    registry.create_document(
        doc_id=doc_id,
        file_hash=file_hash,
        original_filename=source_path.name,
        stored_path=str(managed_path),
        parser_name="docling",
        parser_version=None,
        indexed_status="pending",
        ingested_at=ingested_at,
    )

    try:
        chunk_count = index_service.index_pdf(managed_path, doc_id=doc_id)
        indexed_at = _utc_now_iso()
        registry.mark_indexed(
            doc_id=doc_id,
            chunk_count=chunk_count,
            indexed_at=indexed_at,
        )
    except Exception as exc:
        registry.mark_failed(
            doc_id=doc_id,
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {exc}",
        ) from exc

    created = registry.get_document(doc_id)
    if created is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document was copied but registry record could not be loaded.",
        )

    return DocumentIngestResponse(
        message="Document copied and registered.",
        deduplicated=False,
        document=DocumentSummaryResponse.from_record(created),
    )


@router.post("/documents/{doc_id}/reindex", response_model=DocumentReindexResponse)
def reindex_document(
        doc_id: str,
        registry: DocumentRegistry = Depends(get_document_registry_from_state),
        index_service: IndexService = Depends(get_index_service_from_state),
) -> DocumentReindexResponse:
    existing = registry.get_document(doc_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    registry.mark_pending(doc_id=doc_id)
    try:
        chunk_count = index_service.reindex_document(
            pdf_path=existing.stored_path,
            doc_id=doc_id,
        )
        indexed_at = _utc_now_iso()
        registry.mark_indexed(
            doc_id=doc_id,
            chunk_count=chunk_count,
            indexed_at=indexed_at,
        )
    except Exception as exc:
        registry.mark_failed(
            doc_id=doc_id,
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reindex failed: {exc}",
        ) from exc

    updated = registry.get_document(doc_id)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document was reindexed but could not be reloaded.",
        )

    return DocumentReindexResponse(
        message="Document reindexed.",
        document=DocumentSummaryResponse.from_record(updated),
    )


# Path parameter:
# In "/documents/{doc_id}", the `{doc_id}` part is taken from the URL
# and passed into the function argument named `doc_id`.
# FastAPI matches them by name.
@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
def delete_document(
        doc_id: str,
        registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> DocumentDeleteResponse:
    existing = registry.get_document(doc_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    managed_path = Path(existing.stored_path)
    try:
        managed_path.unlink(missing_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete managed file: {exc}",
        ) from exc

    registry.delete_document(doc_id)

    return DocumentDeleteResponse(
        message="Document deleted from registry.",
        doc_id=doc_id,
    )
