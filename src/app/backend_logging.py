"""Backend file logging helpers for packaged diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

from src.app.paths import AppPaths


def configure_backend_logging(paths: AppPaths) -> Path:
    """Attach one stable backend file handler to the root logger."""
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = paths.backend_log_path.resolve()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    existing_handler: logging.Handler | None = None
    for handler in list(root_logger.handlers):
        if not getattr(handler, "_ldi_backend_file_handler", False):
            continue

        handler_path = getattr(handler, "baseFilename", None)
        if handler_path and Path(handler_path).resolve() == log_path:
            existing_handler = handler
            continue

        root_logger.removeHandler(handler)
        handler.close()

    if existing_handler is None:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler._ldi_backend_file_handler = True  # type: ignore[attr-defined]
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    return log_path
