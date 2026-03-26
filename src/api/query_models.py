"""Pydantic models for query request and response payloads."""

from __future__ import annotations

from pydantic import BaseModel, field_validator
from typing import Literal

class QueryRequest(BaseModel):
    """Request payload for grounded and chat query operations."""

    query: str
    mode: str = "grounded"
    doc_ids: list[str] | None = None
    reasoning_mode: Literal["think", "no_think"] = "no_think"
    stream_thinking: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Normalize and validate the query text."""
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty.")
        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        """Normalize and validate the requested mode."""
        value = value.strip().lower()
        if value not in {"grounded", "chat", "auto"}:
            raise ValueError("mode must be one of: grounded, chat, auto.")
        return value

    @field_validator("doc_ids")
    @classmethod
    def validate_doc_ids(cls, value: list[str] | None) -> list[str] | None:
        """Remove blank document identifiers and collapse empty selections."""
        if value is None:
            return None

        cleaned = [item.strip() for item in value if item and item.strip()]
        return cleaned or None

    @field_validator("reasoning_mode")
    @classmethod
    def validate_reasoning_mode(cls, value: str) -> str:
        """Normalize and validate the reasoning mode."""
        value = value.strip().lower()
        if value not in {"think", "no_think"}:
            raise ValueError("reasoning_mode must be one of: think, no_think.")
        return value

class QueryTimingResponse(BaseModel):
    """Timing breakdown for query execution."""

    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class QuerySourceResponse(BaseModel):
    """One document-level evidence reference returned with a query."""

    doc_id: str | None = None
    original_filename: str | None = None
    pages: list[int]


class QueryResponse(BaseModel):
    """Response payload for non-streaming query execution."""

    query: str
    answer: str
    mode_used: str
    fallback_reason: str | None
    sources: list[QuerySourceResponse]
    used_context_tokens: int
    retrieved_chunk_count: int
    timings: QueryTimingResponse
    reasoning_mode: str
    thinking_content: str | None = None
    thinking_finished: bool = False