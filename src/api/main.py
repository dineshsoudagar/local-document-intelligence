from __future__ import annotations

# Learning:
# This file builds the FastAPI app and prepares app resources at startup.
#
# Prod:
# FastAPI application entrypoint.

from pathlib import Path

from fastapi import FastAPI

from src.api.routes_documents import router as documents_router
from src.api.routes_health import router as health_router
from src.app.paths import AppPaths


def create_app() -> FastAPI:
    """
    Learning:
    Builds and returns the FastAPI application.

    Prod:
    Build and configure the FastAPI application.
    """
    app = FastAPI(
        title="Offline Document Intelligence API",
        version="0.1.0",
    )

    # Learning:
    # Store shared objects on app.state so the whole application can reuse them.
    #
    # Prod:
    # Shared application storage paths.
    app.state.paths = AppPaths(base_dir=Path("storage"))

    @app.on_event("startup")
    def on_startup() -> None:
        """
        Learning:
        This runs once when the server starts.

        Prod:
        Initialize storage directories at application startup.
        """
        # Learning:
        # Create required folders before serving requests.
        #
        # Prod:
        # Ensure storage directories exist.
        app.state.paths.ensure_exists()

    # Learning:
    # Attach route groups to the app.
    #
    # Prod:
    # Register API route modules.
    app.include_router(health_router)
    app.include_router(documents_router)

    return app


app = create_app()