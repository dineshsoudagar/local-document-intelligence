from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.app_state import get_answer_service_from_state
from src.api.query_models import QueryRequest, QueryResponse
from src.generation.answer_service import GroundedAnswerService

router = APIRouter()


def _event_line(event_type: str, data: object) -> bytes:
    return (json.dumps({"type": event_type, "data": data}, ensure_ascii=False) + "\n").encode("utf-8")


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    answer_service: GroundedAnswerService = Depends(get_answer_service_from_state),
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

    return QueryResponse.model_validate(result.to_dict())


@router.post("/query/stream")
def stream_query_documents(
    request: QueryRequest,
    answer_service: GroundedAnswerService = Depends(get_answer_service_from_state),
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
        # Collect streamed chunks so the final event can include the full answer
        answer_parts: list[str] = []

        # Measure generation time separately from retrieval time
        generation_started_at = time.perf_counter()

        # Send initial metadata before token streaming starts
        yield _event_line("start", start_payload.to_dict())

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