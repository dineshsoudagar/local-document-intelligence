"""Domain models for document-level macro routing profiles."""

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
    """
    Normalize the full heading path.

    We never return an empty path because grouping logic must stay stable.
    """
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


@dataclass(slots=True, frozen=True)
class VisualClue:
    """A non-text clue extracted from the document."""

    kind: str
    page_start: int | None = None
    page_end: int | None = None
    caption: str | None = None
    annotations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        page_start, page_end = _normalize_page_range(self.page_start, self.page_end)
        object.__setattr__(self, "kind", _clean_text(self.kind))
        object.__setattr__(self, "page_start", page_start)
        object.__setattr__(self, "page_end", page_end)
        object.__setattr__(self, "caption", _normalize_optional_text(self.caption))
        object.__setattr__(self, "annotations", _dedupe_keep_order(self.annotations))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "kind": self.kind,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "caption": self.caption,
            "annotations": list(self.annotations),
        }


@dataclass(slots=True, frozen=True)
class SectionExcerpt:
    """A representative excerpt selected from a section."""

    chunk_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        page_start, page_end = _normalize_page_range(self.page_start, self.page_end)
        object.__setattr__(self, "chunk_id", _clean_text(self.chunk_id))
        object.__setattr__(self, "text", _clean_text(self.text))
        object.__setattr__(self, "page_start", page_start)
        object.__setattr__(self, "page_end", page_end)
        object.__setattr__(self, "rationale", _normalize_optional_text(self.rationale))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class SectionMacroPacket:
    """
    Pre-summary structured input for one section.

    This is built from parsed chunks before calling the generator.
    """

    doc_id: str
    source_file: str
    heading_path: HeadingPath
    page_start: int | None = None
    page_end: int | None = None
    chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    chunk_count: int = 0
    representative_excerpts: tuple[SectionExcerpt, ...] = field(default_factory=tuple)
    captions: tuple[str, ...] = field(default_factory=tuple)
    annotations: tuple[str, ...] = field(default_factory=tuple)
    visual_clues: tuple[VisualClue, ...] = field(default_factory=tuple)
    block_types: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        page_start, page_end = _normalize_page_range(self.page_start, self.page_end)
        self.doc_id = _clean_text(self.doc_id)
        self.source_file = _clean_text(self.source_file)
        self.heading_path = normalize_heading_path(self.heading_path)
        self.page_start = page_start
        self.page_end = page_end
        self.chunk_ids = _dedupe_keep_order(self.chunk_ids)
        self.chunk_count = self.chunk_count or len(self.chunk_ids)
        self.representative_excerpts = tuple(self.representative_excerpts)
        self.captions = _dedupe_keep_order(self.captions)
        self.annotations = _dedupe_keep_order(self.annotations)
        self.visual_clues = tuple(self.visual_clues)
        self.block_types = _dedupe_keep_order(self.block_types)

    @property
    def section_id(self) -> str:
        """Return the stable section identifier."""
        return build_section_id(self.doc_id, self.heading_path)

    @property
    def display_heading(self) -> str:
        """Return a readable heading path."""
        return " > ".join(self.heading_path)

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return a compact summary prompt payload."""
        return {
            "section_id": self.section_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "heading_path": list(self.heading_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_count": self.chunk_count,
            "chunk_ids": list(self.chunk_ids),
            "captions": list(self.captions),
            "annotations": list(self.annotations),
            "block_types": list(self.block_types),
            "visual_clues": [clue.to_dict() for clue in self.visual_clues],
            "representative_excerpts": [
                excerpt.to_dict() for excerpt in self.representative_excerpts
            ],
        }


@dataclass(slots=True)
class DocumentMacroPacket:
    """
    Pre-summary structured input for the whole document.

    This packet is built after all section packets are available.
    """

    doc_id: str
    source_file: str
    title: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    section_packets: tuple[SectionMacroPacket, ...] = field(default_factory=tuple)
    visual_clues: tuple[VisualClue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.doc_id = _clean_text(self.doc_id)
        self.source_file = _clean_text(self.source_file)
        self.title = _normalize_optional_text(self.title)
        self.page_count = int(self.page_count)
        self.chunk_count = int(self.chunk_count)
        self.section_packets = tuple(self.section_packets)
        self.visual_clues = tuple(self.visual_clues)

    @property
    def heading_paths(self) -> tuple[HeadingPath, ...]:
        """Return section heading paths in document order."""
        return tuple(packet.heading_path for packet in self.section_packets)

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return a compact document-level summary prompt payload."""
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "title": self.title,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "heading_paths": [list(path) for path in self.heading_paths],
            "sections": [packet.to_prompt_payload() for packet in self.section_packets],
            "visual_clues": [clue.to_dict() for clue in self.visual_clues],
        }


@dataclass(slots=True, frozen=True)
class DocumentSectionReference:
    """Compact section reference stored inside the document profile."""

    section_id: str
    heading_path: HeadingPath
    page_start: int | None = None
    page_end: int | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        page_start, page_end = _normalize_page_range(self.page_start, self.page_end)
        object.__setattr__(self, "section_id", _clean_text(self.section_id))
        object.__setattr__(self, "heading_path", normalize_heading_path(self.heading_path))
        object.__setattr__(self, "page_start", page_start)
        object.__setattr__(self, "page_end", page_end)
        object.__setattr__(self, "role", _normalize_optional_text(self.role))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "section_id": self.section_id,
            "heading_path": list(self.heading_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "role": self.role,
        }


@dataclass(slots=True, frozen=True)
class SectionMacroProfile:
    """Final persisted routing profile for one section."""

    section_id: str
    doc_id: str
    source_file: str
    heading_path: HeadingPath
    page_start: int | None
    page_end: int | None
    chunk_count: int
    chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    section_summary: str = ""
    useful_for: tuple[str, ...] = field(default_factory=tuple)
    key_terms: tuple[str, ...] = field(default_factory=tuple)
    entities: tuple[str, ...] = field(default_factory=tuple)
    visual_clues: tuple[VisualClue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        page_start, page_end = _normalize_page_range(self.page_start, self.page_end)
        object.__setattr__(self, "section_id", _clean_text(self.section_id))
        object.__setattr__(self, "doc_id", _clean_text(self.doc_id))
        object.__setattr__(self, "source_file", _clean_text(self.source_file))
        object.__setattr__(self, "heading_path", normalize_heading_path(self.heading_path))
        object.__setattr__(self, "page_start", page_start)
        object.__setattr__(self, "page_end", page_end)
        object.__setattr__(self, "chunk_count", int(self.chunk_count))
        object.__setattr__(self, "chunk_ids", _dedupe_keep_order(self.chunk_ids))
        object.__setattr__(self, "section_summary", _clean_text(self.section_summary))
        object.__setattr__(self, "useful_for", _dedupe_keep_order(self.useful_for))
        object.__setattr__(self, "key_terms", _dedupe_keep_order(self.key_terms))
        object.__setattr__(self, "entities", _dedupe_keep_order(self.entities))
        object.__setattr__(self, "visual_clues", tuple(self.visual_clues))


@dataclass(slots=True, frozen=True)
class DocumentMacroProfile:
    """Final persisted routing profile for one document."""

    doc_id: str
    source_file: str
    title: str | None
    page_count: int
    chunk_count: int
    doc_summary: str
    doc_topics: tuple[str, ...] = field(default_factory=tuple)
    useful_for: tuple[str, ...] = field(default_factory=tuple)
    major_sections: tuple[DocumentSectionReference, ...] = field(default_factory=tuple)
    visual_clues: tuple[VisualClue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _clean_text(self.doc_id))
        object.__setattr__(self, "source_file", _clean_text(self.source_file))
        object.__setattr__(self, "title", _normalize_optional_text(self.title))
        object.__setattr__(self, "page_count", int(self.page_count))
        object.__setattr__(self, "chunk_count", int(self.chunk_count))
        object.__setattr__(self, "doc_summary", _clean_text(self.doc_summary))
        object.__setattr__(self, "doc_topics", _dedupe_keep_order(self.doc_topics))
        object.__setattr__(self, "useful_for", _dedupe_keep_order(self.useful_for))
        object.__setattr__(self, "major_sections", tuple(self.major_sections))
        object.__setattr__(self, "visual_clues", tuple(self.visual_clues))


@dataclass(slots=True, frozen=True)
class MacroPacketBundle:
    """One document packet plus all section packets created from it."""

    document: DocumentMacroPacket
    sections: tuple[SectionMacroPacket, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))