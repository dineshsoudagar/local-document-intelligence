from __future__ import annotations

# Learning:
# This module contains helpers for reading shared objects stored on the FastAPI app.
#
# Prod:
# Helpers for reading shared application state.

from fastapi import Request

from src.app.paths import AppPaths


def get_app_paths_from_state(request: Request) -> AppPaths:
    """
    Learning:
    FastAPI can pass the current HTTP request into this function.

    `request.app` gives access to the running FastAPI application.
    `request.app.state` contains shared objects stored during app setup.

    This helper reads the shared AppPaths object from application state.

    Prod:
    Return application storage paths from application state.
    """
    return request.app.state.paths