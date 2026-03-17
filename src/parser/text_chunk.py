from __future__ import annotations

"""Shared data model for parsed chunks."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ParsedChunk:
    """Represent one parsed unit ready for indexing."""
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk for logging or storage helpers."""
        return asdict(self)
