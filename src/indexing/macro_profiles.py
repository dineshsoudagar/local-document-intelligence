"""Domain models for minimal document and section macro summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

HeadingPath = tuple[str, ...]
UNLABELED_SECTION_TOKEN = "__unlabeled__"


def _clean_text(value: str | None) -> str:
    """Normalize optional text into a stripped string."""
    return str(value or "").strip()


def _dedupe_keep_order(values: Iterable[str]) -> tuple[str, ...]:
    """Remove empty or duplicate strings while preserving order."""
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _clean_text(value)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return tuple(result)


def _normalize_optional_text(value: str | None) -> str | None:
    """Normalize optional text and collapse blanks to None."""
    cleaned = _clean_text(value)
    return cleaned or None


def _normalize_page_range(
    page_start: int | None,
    page_end: int | None,
) -> tuple[int | None, int | None]:
    """Return a stable ascending page range."""
    if page_start is None and page_end is None:
        return None, None
    if page_start is None:
        return page_end, page_end
    if page_end is None:
        return page_start, page_start
    if page_end < page_start:
        return page_end, page_start
    return page_start, page_end


def normalize_heading_path(values: Iterable[str] | None) -> HeadingPath:
    """Normalize the full heading path."""
    if values is None:
        return (UNLABELED_SECTION_TOKEN,)

    normalized = _dedupe_keep_order(values)
    if normalized:
        return normalized

    return (UNLABELED_SECTION_TOKEN,)


def heading_path_key(path: Iterable[str] | None) -> str:
    """Convert a heading path into a stable storage key."""
    return " / ".join(normalize_heading_path(path))


def build_section_id(doc_id: str, heading_path: Iterable[str] | None) -> str:
    """Build a stable section identifier within one document."""
    return f"{_clean_text(doc_id)}::{heading_path_key(heading_path)}"


@dataclass(slots=True)
class SectionMacroPacket:
    """Minimal pre-summary structured input for one section."""

    doc_id: str
    source_file: str
    heading_path: HeadingPath
    section_heading: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section_text: str = ""

    def __post_init__(self) -> None:
        page_start, page_end = _normalize_page_range(self.page_start, self.page_end)
        self.doc_id = _clean_text(self.doc_id)
        self.source_file = _clean_text(self.source_file)
        self.heading_path = normalize_heading_path(self.heading_path)
        self.section_heading = _clean_text(self.section_heading)
        self.page_start = page_start
        self.page_end = page_end
        self.section_text = _clean_text(self.section_text)

    @property
    def section_id(self) -> str:
        return build_section_id(self.doc_id, self.heading_path)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "section_heading": self.section_heading,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_text": self.section_text,
        }


@dataclass(slots=True)
class DocumentMacroPacket:
    """Minimal pre-summary structured input for the whole document."""

    doc_id: str
    source_file: str
    title: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    section_packets: tuple[SectionMacroPacket, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.doc_id = _clean_text(self.doc_id)
        self.source_file = _clean_text(self.source_file)
        self.title = _normalize_optional_text(self.title)
        self.page_count = int(self.page_count)
        self.chunk_count = int(self.chunk_count)
        self.section_packets = tuple(self.section_packets)

    @property
    def heading_paths(self) -> tuple[HeadingPath, ...]:
        """Return section heading paths in document order."""
        return tuple(packet.heading_path for packet in self.section_packets)

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return a minimal document-level summary prompt payload."""
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "title": self.title,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "sections": [packet.to_prompt_payload() for packet in self.section_packets],
        }


@dataclass(slots=True, frozen=True)
class SectionMacroProfile:
    """Final persisted minimal summary for one section."""

    section_id: str
    doc_id: str
    source_file: str
    document_heading: str | None
    section_heading: str
    page_start: int | None
    page_end: int | None
    section_summary: str
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        page_start, page_end = _normalize_page_range(self.page_start, self.page_end)
        object.__setattr__(self, "section_id", _clean_text(self.section_id))
        object.__setattr__(self, "doc_id", _clean_text(self.doc_id))
        object.__setattr__(self, "source_file", _clean_text(self.source_file))
        object.__setattr__(self, "document_heading", _normalize_optional_text(self.document_heading))
        object.__setattr__(self, "section_heading", _clean_text(self.section_heading))
        object.__setattr__(self, "page_start", page_start)
        object.__setattr__(self, "page_end", page_end)
        object.__setattr__(self, "section_summary", _clean_text(self.section_summary))
        object.__setattr__(self, "keywords", _dedupe_keep_order(self.keywords))


@dataclass(slots=True, frozen=True)
class DocumentMacroProfile:
    """Final persisted minimal summary for one document."""

    doc_id: str
    source_file: str
    title: str | None
    page_count: int
    chunk_count: int
    doc_summary: str
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _clean_text(self.doc_id))
        object.__setattr__(self, "source_file", _clean_text(self.source_file))
        object.__setattr__(self, "title", _normalize_optional_text(self.title))
        object.__setattr__(self, "page_count", int(self.page_count))
        object.__setattr__(self, "chunk_count", int(self.chunk_count))
        object.__setattr__(self, "doc_summary", _clean_text(self.doc_summary))
        object.__setattr__(self, "keywords", _dedupe_keep_order(self.keywords))


@dataclass(slots=True, frozen=True)
class MacroPacketBundle:
    """One document packet plus all section packets created from it."""

    document: DocumentMacroPacket
    sections: tuple[SectionMacroPacket, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))