"""Build minimal macro packets from parsed chunks before summary generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import re
from src.indexing.macro_profiles import (
    DocumentMacroPacket,
    HeadingPath,
    MacroPacketBundle,
    SectionMacroPacket,
    UNLABELED_SECTION_TOKEN,
    normalize_heading_path,
)
from src.parser.text_chunk import ParsedChunk


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


def _merge_page_start(left: int | None, right: int | None) -> int | None:
    """Return the earliest known page start."""
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _merge_page_end(left: int | None, right: int | None) -> int | None:
    """Return the latest known page end."""
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _extract_heading_path(chunk: ParsedChunk) -> HeadingPath:
    """Extract the full heading path from chunk metadata."""
    raw = chunk.metadata.get("headings")
    if isinstance(raw, list):
        return normalize_heading_path(str(item) for item in raw)
    return normalize_heading_path(None)


@dataclass(slots=True)
class _SectionAccumulator:
    """Mutable builder-only grouping state for one section."""

    doc_id: str
    source_file: str
    heading_path: HeadingPath
    page_start: int | None = None
    page_end: int | None = None
    chunks: list[ParsedChunk] = field(default_factory=list)

    def add_chunk(self, chunk: ParsedChunk) -> None:
        """Absorb one parsed chunk into the section group."""
        self.chunks.append(chunk)
        self.page_start = _merge_page_start(self.page_start, chunk.page_start)
        self.page_end = _merge_page_end(self.page_end, chunk.page_end)


class MacroPacketBuilder:
    """Convert parsed chunks into minimal document and section macro packets."""

    def __init__(self, *, max_section_chars: int = 4000) -> None:
        self._max_section_chars = max_section_chars

    def build(self, chunks: Sequence[ParsedChunk]) -> MacroPacketBundle:
        """Build one document packet and all section packets for a single document."""
        normalized_chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not normalized_chunks:
            raise ValueError("Cannot build macro packets from empty chunks")

        doc_id = self._resolve_single_doc_id(normalized_chunks)
        source_file = self._resolve_single_source_file(normalized_chunks)

        section_accumulators = self._group_sections(normalized_chunks)
        section_packets = tuple(
            self._build_section_packet(accumulator)
            for accumulator in section_accumulators
        )

        document_packet = self._build_document_packet(
            doc_id=doc_id,
            source_file=source_file,
            chunks=normalized_chunks,
            section_packets=section_packets,
        )

        return MacroPacketBundle(document=document_packet, sections=section_packets)

    def _resolve_single_doc_id(self, chunks: Sequence[ParsedChunk]) -> str:
        """Ensure all chunks belong to one document."""
        doc_ids = {chunk.doc_id for chunk in chunks}
        if len(doc_ids) != 1:
            raise ValueError(f"Expected one doc_id in builder input, got: {sorted(doc_ids)}")
        return next(iter(doc_ids))

    def _resolve_single_source_file(self, chunks: Sequence[ParsedChunk]) -> str:
        """Ensure all chunks belong to one source file."""
        source_files = {chunk.source_file for chunk in chunks}
        if len(source_files) != 1:
            raise ValueError(
                f"Expected one source_file in builder input, got: {sorted(source_files)}"
            )
        return next(iter(source_files))

    def _group_sections(self, chunks: Sequence[ParsedChunk]) -> list[_SectionAccumulator]:
        """Group chunks by full heading path in first-seen document order."""
        grouped: dict[HeadingPath, _SectionAccumulator] = {}

        for chunk in chunks:
            heading_path = _extract_heading_path(chunk)

            if heading_path not in grouped:
                grouped[heading_path] = _SectionAccumulator(
                    doc_id=chunk.doc_id,
                    source_file=chunk.source_file,
                    heading_path=heading_path,
                )

            grouped[heading_path].add_chunk(chunk)

        return list(grouped.values())

    def _build_section_packet(self, accumulator: _SectionAccumulator) -> SectionMacroPacket:
        """Convert one grouped section accumulator into a minimal section packet."""
        heading_parts = [
            value
            for value in accumulator.heading_path
            if value and value != UNLABELED_SECTION_TOKEN
        ]

        section_heading = heading_parts[-1] if heading_parts else "Untitled Section"

        return SectionMacroPacket(
            doc_id=accumulator.doc_id,
            source_file=accumulator.source_file,
            heading_path=accumulator.heading_path,
            section_heading=section_heading,
            page_start=accumulator.page_start,
            page_end=accumulator.page_end,
            section_text=self._build_section_text(accumulator),
        )
    def _build_section_text(self, accumulator: _SectionAccumulator) -> str:
        """Build bounded plain text for one section."""
        parts: list[str] = []
        seen: set[str] = set()

        for chunk in accumulator.chunks:
            text = self._normalize_chunk_text(chunk.text, accumulator.heading_path)
            if not text:
                continue

            key = text.casefold()
            if key in seen:
                continue

            seen.add(key)
            parts.append(text)

        combined = "\n\n".join(parts).strip()

        if len(combined) <= self._max_section_chars:
            return combined

        return combined[: self._max_section_chars].rstrip() + "..."

    def _normalize_chunk_text(self, text: str, heading_path: HeadingPath) -> str:
        """Normalize chunk text and remove repeated heading prefixes."""
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return ""

        cleaned = self._strip_heading_prefixes(normalized, heading_path)
        cleaned = " ".join(cleaned.split()).strip()
        return cleaned

    def _strip_heading_prefixes(self, text: str, heading_path: HeadingPath) -> str:
        """Remove heading strings repeated at the front of contextualized chunk text."""
        cleaned = text

        candidates = sorted(
            [
                heading.strip()
                for heading in heading_path
                if heading and heading != UNLABELED_SECTION_TOKEN
            ],
            key=len,
            reverse=True,
        )

        changed = True
        while changed and cleaned:
            changed = False
            for heading in candidates:
                if not heading:
                    continue

                heading_lower = heading.casefold()
                cleaned_lower = cleaned.casefold()

                if cleaned_lower == heading_lower:
                    cleaned = ""
                    changed = True
                    break

                prefix = f"{heading} "
                if cleaned_lower.startswith(prefix.casefold()):
                    cleaned = cleaned[len(prefix):].strip()
                    changed = True
                    break

        return cleaned

    def _build_document_packet(
        self,
        *,
        doc_id: str,
        source_file: str,
        chunks: Sequence[ParsedChunk],
        section_packets: Sequence[SectionMacroPacket],
    ) -> DocumentMacroPacket:
        """Build the minimal document packet from raw chunks and section packets."""
        max_page = 0

        for chunk in chunks:
            if chunk.page_end is not None:
                max_page = max(max_page, chunk.page_end)
            elif chunk.page_start is not None:
                max_page = max(max_page, chunk.page_start)

        title = self._infer_document_title(
            source_file=source_file,
            section_packets=section_packets,
        )

        return DocumentMacroPacket(
            doc_id=doc_id,
            source_file=source_file,
            title=title,
            page_count=max_page,
            chunk_count=len(chunks),
            section_packets=tuple(section_packets),
        )

    def _infer_document_title(
        self,
        *,
        source_file: str,
        section_packets: Sequence[SectionMacroPacket],
    ) -> str:
        """Infer a document title conservatively."""
        structural_heading_pattern = re.compile(
            r"^((?:[IVXLC]+)\.|(?:[A-Z])\.|(?:\d+(?:\.\d+)*)\.?)\s+"
        )

        for packet in section_packets:
            heading = packet.section_heading.strip()
            if not heading or heading == "Untitled Section":
                continue

            if packet.page_start == 1 and not structural_heading_pattern.match(heading):
                return heading

        return Path(source_file).stem.replace("_", " ").replace("-", " ").strip()