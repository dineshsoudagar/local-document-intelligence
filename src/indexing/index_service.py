from __future__ import annotations

import hashlib
from pathlib import Path

from docling.datamodel.base_models import InputFormat

from src.config.index_config import IndexConfig
from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex


class IndexService:
    """Manage document ingestion and index lifecycle."""

    def __init__(
        self,
        *,
        index: QdrantHybridIndex | None = None,
        index_config: IndexConfig | None = None,
        parser_config: ParserConfig | None = None,
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

    @property
    def index(self) -> QdrantHybridIndex:
        """Expose the underlying index for querying."""
        return self._index

    def ensure_pdf_indexed(self, pdf_path: str | Path) -> str:
        """Index one PDF if it is not already present."""
        path = Path(pdf_path).resolve()
        doc_id = self.build_doc_id(path)

        if not self._index.document_exists(doc_id):
            chunks = self._parser.parse(path, doc_id=doc_id)
            self._index.build(chunks=chunks, rebuild=False)

        return doc_id

    def ingest_pdf(self, pdf_path: str | Path) -> str:
        """Index one PDF if missing and return its doc_id."""
        return self.ensure_pdf_indexed(pdf_path)

    def reindex_pdf(self, pdf_path: str | Path) -> str:
        """Replace one indexed PDF with a freshly parsed version."""
        path = Path(pdf_path).resolve()
        doc_id = self.build_doc_id(path)

        self._index.delete_document(doc_id)
        chunks = self._parser.parse(path, doc_id=doc_id)
        self._index.build(chunks=chunks, rebuild=False)
        return doc_id

    def ingest_folder(self, folder_path: str | Path) -> list[str]:
        """Index all PDFs inside a folder if they are missing."""
        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        doc_ids: list[str] = []
        for pdf_path in sorted(folder.rglob("*.pdf")):
            doc_ids.append(self.ensure_pdf_indexed(pdf_path))
        return doc_ids

    def reindex_folder(self, folder_path: str | Path) -> list[str]:
        """Reindex all PDFs inside a folder."""
        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        doc_ids: list[str] = []
        for pdf_path in sorted(folder.rglob("*.pdf")):
            doc_ids.append(self.reindex_pdf(pdf_path))
        return doc_ids

    def clear(self) -> None:
        """Clear the entire index."""
        self._index.clear()

    @staticmethod
    def build_doc_id(pdf_path: str | Path) -> str:
        """Build a stable document id from file content."""
        path = Path(pdf_path).resolve()
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        stem = path.stem.strip().replace(" ", "_").replace("-", "_").lower()
        return f"{stem}_{digest}"