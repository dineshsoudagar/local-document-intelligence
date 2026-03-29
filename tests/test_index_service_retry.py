"""Unit tests for transient Docling parse retry handling."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from docling.exceptions import ConversionError

from src.indexing.index_service import IndexService
from src.parser.text_chunk import ParsedChunk


def _chunk(doc_id: str) -> ParsedChunk:
    return ParsedChunk(
        chunk_id=f"{doc_id}_chunk_0000",
        doc_id=doc_id,
        source_file="sample.pdf",
        chunk_index=0,
        page_start=1,
        page_end=1,
        text="hello world",
        metadata={"parser": "docling"},
    )


class _StubParser:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[Path, str]] = []

    def parse(self, path: Path, doc_id: str) -> list[ParsedChunk]:
        self.calls.append((Path(path), doc_id))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class IndexServiceParseRetryTests(unittest.TestCase):
    def _service(self, parser: _StubParser) -> IndexService:
        service = IndexService.__new__(IndexService)
        service._parser = parser
        return service

    def _pdf_path(self) -> Path:
        path = Path("storage/documents/doc_396a0fadeb4cd40f.pdf").resolve()
        self.assertTrue(path.is_file(), f"Fixture PDF is missing: {path}")
        return path

    def test_retries_transient_invalid_document_error(self) -> None:
        path = self._pdf_path()
        parser = _StubParser(
            [
                ConversionError("Input document sample.pdf is not valid."),
                [_chunk("doc_123")],
            ]
        )
        service = self._service(parser)

        with patch("src.indexing.index_service.time.sleep") as sleep_mock:
            chunks = service._parse_chunks(path, "doc_123")

        self.assertEqual(chunks[0].doc_id, "doc_123")
        self.assertEqual(len(parser.calls), 2)
        sleep_mock.assert_called_once()

    def test_does_not_retry_other_conversion_errors(self) -> None:
        path = self._pdf_path()
        parser = _StubParser([ConversionError("No pipeline could be initialized.")])
        service = self._service(parser)

        with patch("src.indexing.index_service.time.sleep") as sleep_mock:
            with self.assertRaises(ConversionError):
                service._parse_chunks(path, "doc_123")

        self.assertEqual(len(parser.calls), 1)
        sleep_mock.assert_not_called()

    def test_stops_after_max_invalid_document_retries(self) -> None:
        path = self._pdf_path()
        parser = _StubParser(
            [
                ConversionError("Input document sample.pdf is not valid."),
                ConversionError("Input document sample.pdf is not valid."),
                ConversionError("Input document sample.pdf is not valid."),
            ]
        )
        service = self._service(parser)

        with patch("src.indexing.index_service.time.sleep") as sleep_mock:
            with self.assertRaises(ConversionError):
                service._parse_chunks(path, "doc_123")

        self.assertEqual(len(parser.calls), 3)
        self.assertEqual(sleep_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
