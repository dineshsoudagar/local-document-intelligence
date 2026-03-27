"""Index lifecycle helpers for local document ingestion and reindexing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from docling.datamodel.base_models import InputFormat

from src.config.index_config import IndexConfig
from src.config.parser_config import ParserConfig
from src.indexing.macro_packet_builder import MacroPacketBuilder
from src.indexing.macro_profiles import MacroPacketBundle
from src.parser.docling_parser import DoclingParser
from src.parser.text_chunk import ParsedChunk
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex
from src.utils.io import resolve_pdf_path


class IndexService:
    """Manage document ingestion and index lifecycle."""

    def __init__(
        self,
        *,
        index: QdrantHybridIndex | None = None,
        index_config: IndexConfig | None = None,
        parser_config: ParserConfig | None = None,
        macro_packet_builder: MacroPacketBuilder | None = None,
    ) -> None:
        self._index = index or QdrantHybridIndex(index_config or IndexConfig())
        self._parser = DoclingParser(
            parser_config
            or ParserConfig(
                allowed_formats=[InputFormat.PDF],
                enable_picture_description=False,
                include_picture_chunks=True,
            )
        )
        self._macro_packet_builder = macro_packet_builder or MacroPacketBuilder()

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

    def build_macro_packets(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> MacroPacketBundle:
        """
        Parse one PDF and build macro packets without indexing.

        This is the bridge to the upcoming summary-generation stage.
        It lets you inspect section grouping and packet quality before
        wiring in summarization or storage.
        """
        path = resolve_pdf_path(pdf_path)
        resolved_doc_id = doc_id or self.build_doc_id(path)
        chunks = self._parse_chunks(path, resolved_doc_id)
        return self._macro_packet_builder.build(chunks)

    def parse_with_macro_packets(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> tuple[str, list[ParsedChunk], MacroPacketBundle]:
        """
        Parse one PDF once and return:
        - resolved doc_id
        - parsed chunks
        - macro packet bundle

        This method is the future integration point for:
        parse -> macro summaries -> storage -> chunk indexing
        """
        path = resolve_pdf_path(pdf_path)
        resolved_doc_id = doc_id or self.build_doc_id(path)
        chunks = self._parse_chunks(path, resolved_doc_id)
        packets = self._macro_packet_builder.build(chunks)
        return resolved_doc_id, chunks, packets

    def clear(self) -> None:
        """Clear the entire index."""
        self._index.clear()

    @staticmethod
    def build_doc_id(pdf_path: str | Path) -> str:
        """Build a stable document id from file content."""
        path = resolve_pdf_path(pdf_path)
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        stem = path.stem.strip().replace(" ", "_").replace("-", "_").lower()
        return f"{stem}_{digest}"

    def _parse_chunks(self, pdf_path: str | Path, doc_id: str) -> list[ParsedChunk]:
        """Parse one document into normalized chunks."""
        path = Path(pdf_path).resolve()
        return self._parser.parse(path, doc_id=doc_id)

    def close(self) -> None:
        """Release index resources."""
        self._index.close()