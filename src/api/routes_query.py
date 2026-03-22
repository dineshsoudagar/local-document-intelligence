from typing import cast

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import json
from src.api.app_state import (
    get_answer_service_from_state,
    get_document_registry_from_state,
)
from src.api.query_models import QueryRequest, QueryResponse
from src.app.document_registry import DocumentRegistry
from src.generation.answer_service import GroundedAnswerService

import json
import time
from collections.abc import Iterator
from pathlib import Path
router = APIRouter()


def _event_line(event_type: str, data: object) -> bytes:
    return (json.dumps({"type": event_type, "data": data}, ensure_ascii=False) + "\n").encode("utf-8")


def _attach_original_filenames(
    sources: list[dict[str, object]],
    registry: DocumentRegistry,
) -> list[dict[str, object]]:
    # Cache document lookups so repeated chunks from the same document do not hit SQLite repeatedly
    original_filename_by_doc_id: dict[str, str | None] = {}

    for source in sources:
        doc_id = _resolve_source_doc_id(source)

        # Sources without doc_id cannot be joined to the registry
        if not doc_id:
            source["original_filename"] = None
            continue

        # Load the uploaded filename once per document id
        if doc_id not in original_filename_by_doc_id:
            record = registry.get_document(doc_id)
            original_filename_by_doc_id[doc_id] = (
                record.original_filename if record is not None else None
            )

        # Attach the real uploaded filename for frontend display
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
    """Resolve a registry doc_id from the source payload.

    Older indexed chunks may be missing `doc_id` in their stored payload while still
    exposing the managed filename like `doc_<hash>.pdf` as `source_file`.
    """
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
    answer_service: GroundedAnswerService = Depends(get_answer_service_from_state),
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> QueryResponse:
    try:
        result = answer_service.answer(
            query=request.query,
            mode=request.mode,
            doc_ids=request.doc_ids,
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

    # Convert the answer result into a mutable payload
    payload = result.to_dict()

    # Replace managed storage names with the real uploaded filenames when possible
    payload["sources"] = _collapse_sources_to_references(payload["sources"], registry)

    return QueryResponse.model_validate(payload)


@router.post("/query/stream")
def stream_query_documents(
    request: QueryRequest,
    answer_service: GroundedAnswerService = Depends(get_answer_service_from_state),
    registry: DocumentRegistry = Depends(get_document_registry_from_state),
) -> StreamingResponse:
    # Build the retrieval + generation stream before starting the HTTP response
    try:
        start_payload, token_stream = answer_service.stream(
            query=request.query,
            mode=request.mode,
            doc_ids=request.doc_ids,
        )

    # Invalid user input or unsupported mode becomes HTTP 400
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Any unexpected backend failure becomes HTTP 500
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        ) from exc

    # This generator produces the NDJSON response line by line
    def body() -> Iterator[bytes]:
        # Collect streamed text so the final event can include the full answer
        answer_parts: list[str] = []

        # Measure generation time separately from retrieval time
        generation_started_at = time.perf_counter()

        # Convert the stream start payload into a mutable dictionary
        start_data = start_payload.to_dict()

        # Replace managed storage names with the real uploaded filenames when possible
        start_data["sources"] = _collapse_sources_to_references(start_data["sources"], registry)

        # Send initial metadata before token streaming starts
        yield _event_line("start", start_data)

        try:
            # Stream each generated text piece from the answer service
            for text_piece in token_stream:
                # Ignore empty text pieces if the generator yields them
                if not text_piece:
                    continue

                # Save the text so the complete answer can be reconstructed later
                answer_parts.append(text_piece)

                # Send one streamed token event to the frontend
                yield _event_line("token", {"text": text_piece})

        # If generation fails after streaming started, send an error event in-band
        except Exception as exc:
            yield _event_line("error", {"message": f"Query failed: {exc}"})
            return

        # Compute generation duration once token streaming is complete
        generation_seconds = time.perf_counter() - generation_started_at

        # Send the final event with the full answer and timing breakdown
        yield _event_line(
            "done",
            {
                "answer": "".join(answer_parts),
                "timings": {
                    "retrieval_seconds": start_payload.retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": start_payload.retrieval_seconds + generation_seconds,
                },
            },
        )

    # Return a streaming HTTP response instead of one buffered JSON object
    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )
