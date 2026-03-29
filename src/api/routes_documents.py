"""Document management API routes."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from src.api.app_state import (
    get_app_paths_from_state,
    get_document_registry_from_state,
    get_index_service_from_state,
)
from src.api.document_models import (
    DocumentDeleteResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentReindexResponse,
    DocumentsListResponse,
    DocumentSummaryResponse,
)
from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths

if TYPE_CHECKING:
    from src.indexing.index_service import IndexService


router = APIRouter()
logger = logging.getLogger(__name__)


def _compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_doc_id(file_hash: str) -> str:
    """Derive a stable document id from a file hash."""
    return f"doc_{file_hash[:16]}"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _build_managed_document_path(paths: AppPaths, doc_id: str, suffix: str) -> Path:
    """Return the managed storage path for a document."""
    return paths.documents_dir / f"{doc_id}{suffix.lower()}"


def _next_process_counter(request: Request, attribute_name: str) -> int:
    """Increment and return one per-process request counter stored on app state."""
    current_value = int(getattr(request.app.state, attribute_name, 0)) + 1
    setattr(request.app.state, attribute_name, current_value)
    return current_value


def _index_diagnostic_snapshot(index_service: "IndexService") -> dict[str, str | bool | None]:
    """Return parser warmup diagnostics when the index service exposes them."""
    snapshot_getter = getattr(index_service, "parser_warmup_snapshot", None)
    if callable(snapshot_getter):
        snapshot = snapshot_getter()
        return {
            "parser_warmup_ran_in_process": snapshot.get("parser_warmup_ran_in_process"),
            "parser_warmup_started_at": snapshot.get("parser_warmup_started_at"),
            "parser_warmup_completed_at": snapshot.get("parser_warmup_completed_at"),
            "parser_warmup_completed": snapshot.get("parser_warmup_completed"),
            "parser_warmup_error": snapshot.get("parser_warmup_error"),
        }

    return {
        "parser_warmup_ran_in_process": None,
        "parser_warmup_started_at": None,
        "parser_warmup_completed_at": None,
        "parser_warmup_completed": None,
        "parser_warmup_error": None,
    }


@router.get("/documents", response_model=DocumentsListResponse)
def list_documents(
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> DocumentsListResponse:
    """Return all registered documents ordered by ingest time."""
    return DocumentsListResponse(
        items=[
            DocumentSummaryResponse.from_record(record)
            for record in registry.list_documents()
        ]
    )


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
def ingest_document(
    request: DocumentIngestRequest,
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
    paths: AppPaths = Depends(get_app_paths_from_state),
    index_service: "IndexService" = Depends(get_index_service_from_state),
) -> DocumentIngestResponse:
    """Copy a local PDF into managed storage and index it."""
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
        registry.mark_indexed(
            doc_id=doc_id,
            chunk_count=chunk_count,
            indexed_at=_utc_now_iso(),
        )
    except Exception as exc:
        logger.exception("Document ingest indexing failed for doc_id=%s path=%s", doc_id, managed_path)
        registry.mark_failed(doc_id=doc_id, error_message=str(exc))
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
    index_service: "IndexService" = Depends(get_index_service_from_state),
) -> DocumentReindexResponse:
    """Reindex an already registered document."""
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
        registry.mark_indexed(
            doc_id=doc_id,
            chunk_count=chunk_count,
            indexed_at=_utc_now_iso(),
        )
    except Exception as exc:
        logger.exception("Document reindex failed for doc_id=%s", doc_id)
        registry.mark_failed(doc_id=doc_id, error_message=str(exc))
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


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
def delete_document(
    doc_id: str,
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> DocumentDeleteResponse:
    """Delete a managed document and remove its registry entry."""
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


@router.post("/documents/upload", response_model=DocumentIngestResponse)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
    paths: AppPaths = Depends(get_app_paths_from_state),
    index_service: "IndexService" = Depends(get_index_service_from_state),
) -> DocumentIngestResponse:
    """Upload a PDF, copy it into managed storage, and index it."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    original_filename = file.filename
    suffix = Path(original_filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF uploads are supported right now.",
        )

    paths.documents_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = paths.base_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    managed_path: Path | None = None
    doc_id: str | None = None
    upload_sequence = _next_process_counter(request, "document_upload_count")
    is_first_upload = upload_sequence == 1
    diagnostics = _index_diagnostic_snapshot(index_service)
    file_size_bytes: int | None = None

    logger.info(
        "document_upload_start upload_sequence=%s first_upload=%s filename=%s "
        "parser_warmup_completed=%s parser_warmup_completed_at=%s parser_warmup_error=%s",
        upload_sequence,
        is_first_upload,
        original_filename,
        diagnostics["parser_warmup_completed"],
        diagnostics["parser_warmup_completed_at"],
        diagnostics["parser_warmup_error"],
    )

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            dir=temp_dir,
            delete=False,
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        file_hash = _compute_file_hash(temp_path)
        doc_id = _build_doc_id(file_hash)
        file_size_bytes = temp_path.stat().st_size

        existing = registry.get_document(doc_id)
        if existing is not None:
            temp_path.unlink(missing_ok=True)
            logger.info(
                "document_upload_deduplicated upload_sequence=%s first_upload=%s doc_id=%s "
                "stored_path=%s file_size_bytes=%s",
                upload_sequence,
                is_first_upload,
                doc_id,
                existing.stored_path,
                file_size_bytes,
            )
            return DocumentIngestResponse(
                message="Document already registered.",
                deduplicated=True,
                document=DocumentSummaryResponse.from_record(existing),
            )

        managed_path = _build_managed_document_path(
            paths=paths,
            doc_id=doc_id,
            suffix=suffix,
        )
        temp_path.replace(managed_path)
        file_size_bytes = managed_path.stat().st_size

        logger.info(
            "document_upload_pre_index upload_sequence=%s first_upload=%s doc_id=%s "
            "managed_path=%s file_size_bytes=%s parser_warmup_completed=%s "
            "parser_warmup_ran_in_process=%s",
            upload_sequence,
            is_first_upload,
            doc_id,
            managed_path,
            file_size_bytes,
            diagnostics["parser_warmup_completed"],
            diagnostics["parser_warmup_ran_in_process"],
        )

        registry.create_document(
            doc_id=doc_id,
            file_hash=file_hash,
            original_filename=original_filename,
            stored_path=str(managed_path),
            parser_name="docling",
            parser_version=None,
            indexed_status="pending",
            ingested_at=_utc_now_iso(),
        )
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()

    try:
        chunk_count = index_service.index_pdf(managed_path, doc_id=doc_id)
        registry.mark_indexed(
            doc_id=doc_id,
            chunk_count=chunk_count,
            indexed_at=_utc_now_iso(),
        )
        logger.info(
            "document_upload_success upload_sequence=%s first_upload=%s doc_id=%s "
            "managed_path=%s file_size_bytes=%s chunk_count=%s parser_warmup_completed=%s",
            upload_sequence,
            is_first_upload,
            doc_id,
            managed_path,
            file_size_bytes,
            chunk_count,
            diagnostics["parser_warmup_completed"],
        )
    except Exception as exc:
        logger.exception(
            "document_upload_indexing_failed upload_sequence=%s first_upload=%s doc_id=%s "
            "managed_path=%s file_size_bytes=%s parser_warmup_completed=%s "
            "parser_warmup_error=%s error=%s",
            upload_sequence,
            is_first_upload,
            doc_id,
            managed_path,
            file_size_bytes,
            diagnostics["parser_warmup_completed"],
            diagnostics["parser_warmup_error"],
            exc,
        )
        registry.mark_failed(doc_id=doc_id, error_message=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload succeeded but indexing failed: {exc}",
        ) from exc

    created = registry.get_document(doc_id)
    if created is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document uploaded and indexed, but registry reload failed.",
        )

    return DocumentIngestResponse(
        message="Document uploaded and indexed.",
        deduplicated=False,
        document=DocumentSummaryResponse.from_record(created),
    )
