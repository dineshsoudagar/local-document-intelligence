from __future__ import annotations

from pathlib import Path

from src.indexing.index_service import IndexService


class _FlakyParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, path: Path, *, doc_id: str) -> list[object]:
        self.calls += 1
        if self.calls == 1:
            raise ValueError(f"Input document {path.name} is not valid.")
        return [{"doc_id": doc_id, "path": str(path)}]


class _BrokenParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, path: Path, *, doc_id: str) -> list[object]:
        self.calls += 1
        raise ValueError("Synthetic permanent parser failure")


class _WarmupParser:
    def __init__(self) -> None:
        self.calls = 0
        self.paths: list[Path] = []

    def parse(self, path: Path, *, doc_id: str) -> list[object]:
        self.calls += 1
        self.paths.append(path)
        return []


def _build_service(parser: object) -> IndexService:
    service = object.__new__(IndexService)
    service._parser = parser
    service._parser_warmed_up = False
    return service


def test_parse_chunks_retries_transient_docling_validation_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

    parser = _FlakyParser()
    service = _build_service(parser)

    chunks = service._parse_chunks(pdf_path, "doc_retry")

    assert parser.calls == 2
    assert chunks == [{"doc_id": "doc_retry", "path": str(pdf_path.resolve())}]


def test_parse_chunks_does_not_retry_permanent_parser_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

    parser = _BrokenParser()
    service = _build_service(parser)

    try:
        service._parse_chunks(pdf_path, "doc_fail")
    except ValueError as exc:
        assert str(exc) == "Synthetic permanent parser failure"
    else:
        raise AssertionError("Expected ValueError to be raised")

    assert parser.calls == 1


def test_warm_up_parser_primes_once() -> None:
    parser = _WarmupParser()
    service = _build_service(parser)

    service.warm_up_parser()
    service.warm_up_parser()

    assert parser.calls == 1
    assert service._parser_warmed_up is True
    assert parser.paths[0].suffix.lower() == ".pdf"
