"""Index lifecycle helpers for local document ingestion and reindexing."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from pypdf import PdfWriter

from src.config.index_config import IndexConfig
from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser
from src.parser.text_chunk import ParsedChunk
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex
from src.utils.io import resolve_pdf_path
from src.config.generator_config import GeneratorConfig

class IndexService:
    """Manage document ingestion and index lifecycle."""

    _PARSE_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.35, 0.9)

    def __init__(
        self,
        *,
        index: QdrantHybridIndex | None = None,
        index_config: IndexConfig | None = None,
        parser_config: ParserConfig | None = None,
        generator_config: GeneratorConfig | None = None,
    ) -> None:
        resolved_index_config = index_config or IndexConfig()
        resolved_parser_config = (
            parser_config
            or ParserConfig(
                allowed_formats=[InputFormat.PDF],
                enable_picture_description=False,
                include_picture_chunks=True,
            )
        )
        resolved_generator_config = generator_config or GeneratorConfig(
            project_root=resolved_index_config.project_root,
            model_catalog=resolved_index_config.model_catalog,
            pipeline_models=resolved_index_config.pipeline_models,
        )

        self._index = index or QdrantHybridIndex(resolved_index_config)
        self._parser = DoclingParser(
            resolved_parser_config
        )
        self._generator_config = resolved_generator_config
        self._parser_warmed_up = False

    @property
    def index(self) -> QdrantHybridIndex:
        """Expose the underlying index for querying."""
        return self._index

    def ensure_pdf_indexed(self, pdf_path: str | Path) -> str:
        """Index one PDF if it is not already present."""
        path = Path(pdf_path).resolve()
        doc_id = self.build_doc_id(path)

        if not self._index.document_exists(doc_id):
            chunks = self._parse_chunks(path, doc_id)
            self._index.build(chunks=chunks, rebuild=False)

        return doc_id

    def index_pdf(self, pdf_path: str | Path, doc_id: str) -> int:
        """Parse and index one document under the supplied doc_id."""
        path = Path(pdf_path).resolve()
        chunks = self._parse_chunks(path, doc_id)
        self._index.build(chunks=chunks, rebuild=False)
        return len(chunks)

    def reindex_document(self, pdf_path: str | Path, doc_id: str) -> int:
        """Delete and rebuild one indexed document."""
        path = Path(pdf_path).resolve()
        self._index.delete_document(doc_id)
        chunks = self._parse_chunks(path, doc_id)
        self._index.build(chunks=chunks, rebuild=False)
        return len(chunks)

    def ingest_pdf(self, pdf_path: str | Path) -> str:
        """Index one PDF if missing and return its doc_id."""
        return self.ensure_pdf_indexed(pdf_path)

    def reindex_pdf(self, pdf_path: str | Path) -> str:
        """Replace one indexed PDF with a freshly parsed version."""
        path = resolve_pdf_path(pdf_path)
        doc_id = self.build_doc_id(path)

        self._index.delete_document(doc_id)
        chunks = self._parse_chunks(path, doc_id)
        self._index.build(chunks=chunks, rebuild=False)
        return doc_id


    def clear(self) -> None:
        """Clear the entire index."""
        self._index.clear()

    def warm_up_parser(self) -> None:
        """Prime lazy parser dependencies so the first real upload is not a cold start."""
        if self._parser_warmed_up:
            return

        fd, temp_path_str = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        temp_path = Path(temp_path_str)

        try:
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            with temp_path.open("wb") as file_handle:
                writer.write(file_handle)

            self._parse_chunks(temp_path, "__warmup__")
            self._parser_warmed_up = True
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def build_doc_id(pdf_path: str | Path) -> str:
        """Build a stable document id from file content."""
        path = resolve_pdf_path(pdf_path)
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        stem = path.stem.strip().replace(" ", "_").replace("-", "_").lower()
        return f"{stem}_{digest}"

    def _parse_chunks(self, pdf_path: str | Path, doc_id: str) -> list[ParsedChunk]:
        """Parse one document into normalized chunks."""
        path = resolve_pdf_path(pdf_path)
        attempts = 1 + len(self._PARSE_RETRY_DELAYS_SECONDS)

        for attempt_index in range(attempts):
            try:
                return self._parser.parse(path, doc_id=doc_id)
            except Exception as exc:
                if not self._should_retry_parse(exc, attempt_index=attempt_index):
                    raise
                time.sleep(self._PARSE_RETRY_DELAYS_SECONDS[attempt_index])

        raise RuntimeError(f"Failed to parse document after {attempts} attempts: {path}")

    def _should_retry_parse(self, exc: Exception, *, attempt_index: int) -> bool:
        """Return whether the parse failure looks transient enough to retry once more."""
        if attempt_index >= len(self._PARSE_RETRY_DELAYS_SECONDS):
            return False

        if isinstance(exc, PermissionError):
            return True

        message = str(exc).lower()
        transient_markers = (
            "input document",
            "is not valid",
            "permission denied",
            "temporarily unavailable",
            "resource busy",
        )
        return any(marker in message for marker in transient_markers)

    def close(self) -> None:
        """Release index resources."""
        self._index.close()
