from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

from src.config.parser_config import ParserConfig
from .text_chunk import ParsedChunk
from src.parser.text_cleaner import TextCleaner


class PDFParser:
    def __init__(self, config: ParserConfig) -> None:
        config.validate()
        self._config = config
        self._reader = PyMuPDFReader()
        self._splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def load_pdf_pages(self, pdf_path: str | Path) -> list[Document]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        return self._reader.load_data(file_path=str(path))

    @staticmethod
    def _resolve_page_range(
            start_char: int,
            end_char: int,
            page_ranges: list[dict[str, int]],
    ) -> tuple[int | None, int | None]:
        if start_char < 0 or end_char < 0 or not page_ranges:
            return None, None

        page_start = None
        page_end = None

        for page in page_ranges:
            if page["start_char"] <= start_char <= page["end_char"]:
                page_start = page["page_num"]
            if page["start_char"] <= end_char <= page["end_char"]:
                page_end = page["page_num"]

        if page_start is None:
            for page in page_ranges:
                if start_char < page["start_char"]:
                    page_start = page["page_num"]
                    break

        if page_end is None:
            for page in reversed(page_ranges):
                if end_char > page["end_char"]:
                    page_end = page["page_num"]
                    break

        return page_start, page_end

    def build_document(
            self,
            pdf_path: str | Path,
            doc_id: str | None = None,
    ) -> tuple[Document, list[dict[str, int]]]:
        path = Path(pdf_path)
        page_docs = self.load_pdf_pages(path)
        resolved_doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"

        cleaned_pages: list[str] = []
        page_ranges: list[dict[str, int]] = []

        separator = "\n\n"
        cursor = 0

        for index, page_doc in enumerate(page_docs):
            cleaned_text = TextCleaner.clean(page_doc.text or "")
            if not cleaned_text:
                continue

            page_number = int(page_doc.metadata.get("page", index + 1))
            start_char = cursor
            end_char = start_char + len(cleaned_text) - 1

            cleaned_pages.append(cleaned_text)
            page_ranges.append(
                {
                    "page_num": page_number,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

            cursor = end_char + 1 + len(separator)

        full_text = separator.join(cleaned_pages)

        metadata = {
            "doc_id": resolved_doc_id,
            "source_file": path.name,
        }

        document = Document(text=full_text, metadata=metadata)
        return document, page_ranges

    def parse(
            self,
            pdf_path: str | Path,
            doc_id: str | None = None,
    ) -> list[ParsedChunk]:
        document, page_ranges = self.build_document(pdf_path=pdf_path, doc_id=doc_id)
        nodes = self._splitter.get_nodes_from_documents([document])

        chunks: list[ParsedChunk] = []
        source_path = Path(pdf_path)
        search_cursor = 0

        for index, node in enumerate(nodes):
            metadata = dict(node.metadata) if node.metadata else {}
            resolved_doc_id = metadata.get("doc_id", "unknown_doc")
            source_file = metadata.get("source_file", source_path.name)

            chunk_text = node.text
            start_char = document.text.find(chunk_text, search_cursor)

            if start_char == -1:
                start_char = document.text.find(chunk_text)

            end_char = start_char + len(chunk_text) - 1 if start_char != -1 else -1

            if start_char != -1:
                search_cursor = start_char + 1

            page_start, page_end = self._resolve_page_range(
                start_char=start_char,
                end_char=end_char,
                page_ranges=page_ranges,
            )

            chunk_metadata = {
                "start_char": start_char,
                "end_char": end_char,
            }

            chunks.append(
                ParsedChunk(
                    chunk_id=f"{resolved_doc_id}_chunk_{index:04d}",
                    doc_id=resolved_doc_id,
                    source_file=source_file,
                    chunk_index=index,
                    page_start=page_start,
                    page_end=page_end,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

        return chunks
