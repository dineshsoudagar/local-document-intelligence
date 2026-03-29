"""Index lifecycle helpers for local document ingestion and reindexing."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
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

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


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
        self._parser_warmup_ran_in_process = False
        self._parser_warmup_started_at: str | None = None
        self._parser_warmup_completed_at: str | None = None
        self._parser_warmup_error: str | None = None

    @property
    def index(self) -> QdrantHybridIndex:
        """Expose the underlying index for querying."""
        return self._index

    @property
    def parser_warmup_ran_in_process(self) -> bool:
        """Return whether parser warmup was attempted in the current process."""
        return self._parser_warmup_ran_in_process

    @property
    def parser_warmup_started_at(self) -> str | None:
        """Return the parser warmup start time."""
        return self._parser_warmup_started_at

    @property
    def parser_warmup_completed_at(self) -> str | None:
        """Return the parser warmup completion time."""
        return self._parser_warmup_completed_at

    @property
    def parser_warmup_error(self) -> str | None:
        """Return the last parser warmup error, if any."""
        return self._parser_warmup_error

    @property
    def parser_warmup_completed(self) -> bool:
        """Return whether parser warmup completed successfully."""
        return self._parser_warmed_up

    def parser_warmup_snapshot(self) -> dict[str, str | bool | None]:
        """Return parser warmup diagnostics for health/debug endpoints."""
        return {
            "parser_warmup_ran_in_process": self.parser_warmup_ran_in_process,
            "parser_warmup_started_at": self.parser_warmup_started_at,
            "parser_warmup_completed_at": self.parser_warmup_completed_at,
            "parser_warmup_completed": self.parser_warmup_completed,
            "parser_warmup_error": self.parser_warmup_error,
        }

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
            logger.info(
                "parser_warmup_skipped already_warmed_up=true completed_at=%s",
                self._parser_warmup_completed_at,
            )
            return

        fd, temp_path_str = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        temp_path = Path(temp_path_str)
        self._parser_warmup_ran_in_process = True
        self._parser_warmup_started_at = _utc_now_iso()
        self._parser_warmup_error = None
        logger.info(
            "parser_warmup_start path=%s started_at=%s",
            temp_path,
            self._parser_warmup_started_at,
        )

        try:
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            with temp_path.open("wb") as file_handle:
                writer.write(file_handle)

            self._parse_chunks(temp_path, "__warmup__")
            self._parser_warmed_up = True
            self._parser_warmup_completed_at = _utc_now_iso()
            logger.info(
                "parser_warmup_complete path=%s completed_at=%s",
                temp_path,
                self._parser_warmup_completed_at,
            )
        except Exception as exc:
            self._parser_warmup_error = str(exc)
            self._parser_warmup_completed_at = None
            logger.exception(
                "parser_warmup_failed path=%s error=%s",
                temp_path,
                exc,
            )
            raise
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                logger.warning(
                    "parser_warmup_temp_cleanup_failed path=%s error=%s",
                    temp_path,
                    cleanup_exc,
                )

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
        is_warmup = doc_id == "__warmup__"
        logger.info(
            "parse_chunks_start doc_id=%s path=%s is_warmup=%s parser_warmup_completed=%s",
            doc_id,
            path,
            is_warmup,
            self.parser_warmup_completed,
        )

        for attempt_index in range(attempts):
            attempt_number = attempt_index + 1
            try:
                chunks = self._parser.parse(path, doc_id=doc_id)
                if attempt_number > 1:
                    logger.info(
                        "parse_chunks_success_after_retry doc_id=%s path=%s attempt=%s",
                        doc_id,
                        path,
                        attempt_number,
                    )
                return chunks
            except Exception as exc:
                should_retry = self._should_retry_parse(exc, attempt_index=attempt_index)
                if not should_retry:
                    logger.exception(
                        "parse_chunks_failed doc_id=%s path=%s attempt=%s final_error=%s",
                        doc_id,
                        path,
                        attempt_number,
                        exc,
                    )
                    raise
                retry_delay = self._PARSE_RETRY_DELAYS_SECONDS[attempt_index]
                logger.warning(
                    "parse_chunks_retry doc_id=%s path=%s attempt=%s error=%s delay_seconds=%.2f",
                    doc_id,
                    path,
                    attempt_number,
                    exc,
                    retry_delay,
                )
                time.sleep(retry_delay)

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
