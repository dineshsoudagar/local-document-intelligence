"""Query API routes for buffered and streaming answer generation."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.app_state import (
    get_answer_service_from_state,
    get_document_registry_from_state,
)
from src.api.query_models import QueryRequest, QueryResponse
from src.app.document_registry import DocumentRegistry

if TYPE_CHECKING:
    from src.generation.answer_service import GroundedAnswerService


router = APIRouter()


def _event_line(event_type: str, data: object) -> bytes:
    """Serialize one NDJSON event line for the streaming response."""
    return (
        json.dumps({"type": event_type, "data": data}, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _attach_original_filenames(
    sources: list[dict[str, object]],
    registry: DocumentRegistry,
) -> list[dict[str, object]]:
    """Attach uploaded filenames to raw source payloads when a registry match exists."""
    original_filename_by_doc_id: dict[str, str | None] = {}

    for source in sources:
        doc_id = _resolve_source_doc_id(source)
        if not doc_id:
            source["original_filename"] = None
            continue

        if doc_id not in original_filename_by_doc_id:
            record = registry.get_document(doc_id)
            original_filename_by_doc_id[doc_id] = (
                record.original_filename if record is not None else None
            )

        source["original_filename"] = original_filename_by_doc_id[doc_id]

    return sources


def _expand_pages(source: dict[str, object]) -> list[int]:
    """Return a deduplicated page list for one source item."""
    page_start_value = source.get("page_start")
    page_end_value = source.get("page_end")

    if page_start_value is None and page_end_value is None:
        return []
    if page_start_value is None:
        page_start_value = page_end_value
    if page_end_value is None:
        page_end_value = page_start_value

    try:
        page_start = int(str(page_start_value))
        page_end = int(str(page_end_value))
    except (TypeError, ValueError):
        return []

    if page_end < page_start:
        page_start, page_end = page_end, page_start

    return list(range(page_start, page_end + 1))


def _collapse_sources_to_references(
    sources: list[dict[str, object]],
    registry: DocumentRegistry,
) -> list[dict[str, object]]:
    """Collapse chunk-level sources into document-level references."""
    normalized_sources = _attach_original_filenames(sources, registry)
    references_by_key: dict[str, dict[str, object]] = {}

    for index, source in enumerate(normalized_sources):
        doc_id = _resolve_source_doc_id(source)
        original_filename = source.get("original_filename")
        source_file = source.get("source_file")
        reference_key = str(doc_id or original_filename or source_file or "unknown")

        if reference_key not in references_by_key:
            references_by_key[reference_key] = {
                "order": index,
                "doc_id": doc_id,
                "original_filename": original_filename,
                "pages": set(),
            }

        reference = references_by_key[reference_key]
        cast(set[int], reference["pages"]).update(_expand_pages(source))

    collapsed_references: list[dict[str, object]] = []
    for reference in references_by_key.values():
        collapsed_references.append(
            {
                "doc_id": reference["doc_id"],
                "original_filename": reference["original_filename"],
                "pages": sorted(cast(set[int], reference["pages"])),
                "order": reference["order"],
            }
        )

    collapsed_references.sort(
        key=lambda reference: (
            cast(int, reference["order"]),
            str(reference["original_filename"] or ""),
        )
    )
    for reference in collapsed_references:
        reference.pop("order", None)
    return collapsed_references


def _resolve_source_doc_id(source: dict[str, object]) -> str | None:
    """Resolve a registry doc_id from a source payload."""
    doc_id_value = source.get("doc_id")
    if doc_id_value:
        return str(doc_id_value)

    source_file_value = source.get("source_file")
    if not source_file_value:
        return None

    source_stem = Path(str(source_file_value)).stem
    if source_stem.startswith("doc_"):
        return source_stem

    return None


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    answer_service: "GroundedAnswerService" = Depends(get_answer_service_from_state),
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> QueryResponse:
    """Run a non-streaming grounded query and return the final answer payload."""
    try:
        result = answer_service.answer(
            query=request.query,
            mode=request.mode,
            doc_ids=request.doc_ids,
            reasoning_mode=request.reasoning_mode,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc

    payload = result.to_dict()
    payload["sources"] = _collapse_sources_to_references(payload["sources"], registry)
    return QueryResponse.model_validate(payload)


@router.post("/query/stream")
def stream_query_documents(
    request: QueryRequest,
    answer_service: "GroundedAnswerService" = Depends(get_answer_service_from_state),
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> StreamingResponse:
    """Stream query output as NDJSON events."""
    print(f"Received query: {request.query}, mode: {request.mode}, doc_ids: {request.doc_ids}, reasoning_mode: {request.reasoning_mode}, stream_thinking: {request.stream_thinking}")  # Debug log      
    try:
        start_payload, event_stream = answer_service.stream(
            query=request.query,
            mode=request.mode,
            doc_ids=request.doc_ids,
            reasoning_mode=request.reasoning_mode,
            stream_thinking=request.stream_thinking,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc

    def body() -> Iterator[bytes]:
        """Yield the NDJSON event stream for one query."""
        thinking_parts: list[str] = []
        answer_parts: list[str] = []
        thinking_finished = request.reasoning_mode == "no_think"
        generation_started_at = time.perf_counter()

        start_data = start_payload.to_dict()
        start_data["sources"] = _collapse_sources_to_references(
            start_data["sources"],
            registry,
        )
        yield _event_line("start", start_data)

        try:
            for event in event_stream:
                if event.kind == "thinking_token":
                    if event.text:
                        thinking_parts.append(event.text)
                        yield _event_line("thinking_token", {"text": event.text})
                    continue

                if event.kind == "thinking_done":
                    thinking_finished = True
                    yield _event_line("thinking_done", {})
                    continue

                if event.kind == "answer_token":
                    if event.text:
                        answer_parts.append(event.text)
                        yield _event_line("answer_token", {"text": event.text})
                    continue

        except Exception as exc:
            yield _event_line("error", {"message": f"Query failed: {exc}"})
            return

        generation_seconds = time.perf_counter() - generation_started_at
        yield _event_line(
            "done",
            {
                "answer": "".join(answer_parts),
                "thinking_content": "".join(thinking_parts) or None,
                "thinking_finished": thinking_finished,
                "reasoning_mode": request.reasoning_mode,
                "timings": {
                    "retrieval_seconds": start_payload.retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": start_payload.retrieval_seconds + generation_seconds,
                },
            },
        )

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )
