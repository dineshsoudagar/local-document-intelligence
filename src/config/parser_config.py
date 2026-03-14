from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParserConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")