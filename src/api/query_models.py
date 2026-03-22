from __future__ import annotations

from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    query: str
    mode: str = "grounded"
    doc_ids: list[str] | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty.")
        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"grounded", "chat", "auto"}:
            raise ValueError("mode must be one of: grounded, chat, auto.")
        return value

    @field_validator("doc_ids")
    @classmethod
    def validate_doc_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        cleaned = [item.strip() for item in value if item and item.strip()]
        return cleaned or None

class QueryTimingResponse(BaseModel):
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class QuerySourceResponse(BaseModel):
    doc_id: str | None = None
    original_filename: str | None = None
    pages: list[int]

class QueryResponse(BaseModel):
    query: str
    answer: str
    mode_used: str
    fallback_reason: str | None
    sources: list[QuerySourceResponse]
    used_context_tokens: int
    retrieved_chunk_count: int
    timings: QueryTimingResponse
