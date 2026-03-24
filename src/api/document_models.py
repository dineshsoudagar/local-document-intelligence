"""Pydantic models for document-related API routes."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from src.app.document_registry import DocumentRecord


class DocumentIngestRequest(BaseModel):
    """Request payload for file-based document ingest operations."""

    source_path: str

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        """Normalize and validate the source path field."""
        value = value.strip()
        if not value:
            raise ValueError("source_path must not be empty.")
        return value


class DocumentSummaryResponse(BaseModel):
    """Serialized document metadata returned by the API."""

    doc_id: str
    original_filename: str
    stored_path: str
    parser_name: str
    parser_version: str | None
    indexed_status: str
    chunk_count: int | None
    ingested_at: str
    indexed_at: str | None
    last_error: str | None

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "DocumentSummaryResponse":
        """Build a response model from a persisted registry record."""
        return cls(
            doc_id=record.doc_id,
            original_filename=record.original_filename,
            stored_path=record.stored_path,
            parser_name=record.parser_name,
            parser_version=record.parser_version,
            indexed_status=record.indexed_status,
            chunk_count=record.chunk_count,
            ingested_at=record.ingested_at,
            indexed_at=record.indexed_at,
            last_error=record.last_error,
        )


class DocumentsListResponse(BaseModel):
    """Response payload for listing documents."""

    items: list[DocumentSummaryResponse]


class DocumentIngestResponse(BaseModel):
    """Response payload for ingest and upload operations."""

    message: str
    deduplicated: bool
    document: DocumentSummaryResponse


class DocumentDeleteResponse(BaseModel):
    """Response payload for document deletion."""

    message: str
    doc_id: str


class DocumentReindexResponse(BaseModel):
    """Response payload for document reindexing."""

    message: str
    document: DocumentSummaryResponse
