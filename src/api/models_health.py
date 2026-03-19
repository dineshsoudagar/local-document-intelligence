from __future__ import annotations

# Learning:
# This module defines response models for health routes.
#
# Prod:
# Health API models.

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """
    Learning:
    A Pydantic model defines the shape of data.

    Here we say a health response must contain:
    - status
    - documents_dir

    FastAPI uses this for validation and API docs.

    Prod:
    Response model for the health endpoint.
    """

    status: str
    documents_dir: str