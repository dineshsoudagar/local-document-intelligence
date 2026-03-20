from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.app_state import get_answer_service_from_state
from src.api.query_models import QueryRequest, QueryResponse
from src.generation.answer_service import GroundedAnswerService

router = APIRouter()


# Learning:
# This route takes a JSON request body (`request: QueryRequest`)
# and also receives a shared backend service through dependency injection.
# FastAPI validates the input body and the output response model automatically.
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
