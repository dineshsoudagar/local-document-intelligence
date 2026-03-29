"""Small filesystem helpers shared by CLI utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: Any, output_path: str | Path) -> None:
    """Write JSON data to disk with UTF-8 encoding and parent creation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def resolve_pdf_path(value: str | Path) -> Path:
    """Resolve one existing PDF file path."""
    path = Path(value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported: {path}")
    return path
