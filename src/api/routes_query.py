from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.app_state import (
    get_answer_service_from_state,
    get_document_registry_from_state,
)
from src.api.query_models import QueryRequest, QueryResponse
from src.app.document_registry import DocumentRegistry
from src.generation.answer_service import GroundedAnswerService

router = APIRouter()


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

    payload = result.to_dict()

    # Map doc_id -> original uploaded filename from the registry.
    original_filename_by_doc_id: dict[str, str | None] = {}

    for source in payload["sources"]:
        doc_id = source.get("doc_id")
        if not doc_id:
            source["original_filename"] = None
            continue

        if doc_id not in original_filename_by_doc_id:
            record = registry.get_document(doc_id)
            original_filename_by_doc_id[doc_id] = (
                record.original_filename if record is not None else None
            )

        source["original_filename"] = original_filename_by_doc_id[doc_id]

    return QueryResponse.model_validate(payload)
