from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from src.config.parser_config import ParserConfig
from src.parser.text_chunk import ParsedChunk


class DoclingParser:
    def __init__(self, config: ParserConfig) -> None:
        config.validate()
        self._config = config
        self._converter = DocumentConverter()
        self._chunker = HybridChunker()

    def parse(self, pdf_path: str | Path, doc_id: str | None = None) -> list[ParsedChunk]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        resolved_doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"

        result = self._converter.convert(str(path))
        document = result.document

        chunks: list[ParsedChunk] = []

        for chunk_index, chunk in enumerate(self._chunker.chunk(dl_doc=document)):
            text = self._chunker.contextualize(chunk=chunk).strip()
            if not text:
                continue

            page_start, page_end = self._extract_chunk_page_range(chunk)

            metadata: dict[str, Any] = {
                "parser": "docling_hybrid",
                "block_type": "text",
            }

            headings = getattr(getattr(chunk, "meta", None), "headings", None)
            if headings:
                metadata["headings"] = headings

            chunks.append(
                ParsedChunk(
                    chunk_id=f"{resolved_doc_id}_chunk_{chunk_index:04d}",
                    doc_id=resolved_doc_id,
                    source_file=path.name,
                    chunk_index=chunk_index,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _extract_chunk_page_range(chunk: Any) -> tuple[int | None, int | None]:
        page_numbers: list[int] = []

        meta = getattr(chunk, "meta", None)
        if meta is None:
            return None, None

        doc_items = getattr(meta, "doc_items", None) or []
        for item in doc_items:
            prov = getattr(item, "prov", None) or []
            for prov_item in prov:
                page_no = getattr(prov_item, "page_no", None)
                if isinstance(page_no, int):
                    page_numbers.append(page_no)

        if not page_numbers:
            return None, None

        return min(page_numbers), max(page_numbers)