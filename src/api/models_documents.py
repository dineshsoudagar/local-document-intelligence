from __future__ import annotations

# Learning:
# This module defines request and response models for document routes.
#
# Prod:
# Document API models.

from pydantic import BaseModel, field_validator


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


class DocumentIngestResponse(BaseModel):
    """
    Learning:
    This model defines the JSON returned by the ingest endpoint.

    Prod:
    Response model for document ingest operations.
    """

    message: str
    source_path: str