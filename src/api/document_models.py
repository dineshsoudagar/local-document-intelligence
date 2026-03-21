from __future__ import annotations

# Learning:
# This module defines request and response models for document routes.
#
# Prod:
# Document API models.

from pydantic import BaseModel, field_validator
from src.app.document_registry import DocumentRecord


class DocumentIngestRequest(BaseModel):
    """
    Learning:
    This model validates only the request shape and basic text cleanup.

    Prod:
    Request model for document ingest operations.
    """

    source_path: str

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        """
        Learning:
        Keep request-model validation simple.
        Only reject blank input here.

        Prod:
        Normalize and validate the source path field.
        """
        # Learning:
        # Strip whitespace so values like "   " are rejected.
        #
        # Prod:
        # Normalize incoming path text.
        value = value.strip()

        if not value:
            raise ValueError("source_path must not be empty.")

        return value


class DocumentSummaryResponse(BaseModel):
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
    items: list[DocumentSummaryResponse]


class DocumentIngestResponse(BaseModel):
    message: str
    deduplicated: bool
    document: DocumentSummaryResponse


class DocumentDeleteResponse(BaseModel):
    message: str
    doc_id: str


class DocumentReindexResponse(BaseModel):
    message: str
    document: DocumentSummaryResponse
