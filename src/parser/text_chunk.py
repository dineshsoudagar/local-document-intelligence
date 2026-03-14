from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ParsedChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
