"""Build macro packets from parsed chunks before summary generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from src.indexing.macro_profiles import (
    HeadingPath,
    MacroPacketBundle,
    SectionExcerpt,
    SectionMacroPacket,
    DocumentMacroPacket,
    VisualClue,
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
    """
    Extract the full heading path from chunk metadata.

    This must preserve the full hierarchy, not just the last heading.
    """
    raw = chunk.metadata.get("headings")
    if isinstance(raw, list):
        return normalize_heading_path(str(item) for item in raw)
    return normalize_heading_path(None)


def _extract_block_type(chunk: ParsedChunk) -> str:
    """Return stable block type for later routing/debug use."""
    return _clean_text(str(chunk.metadata.get("block_type") or "text")) or "text"


def _extract_chunk_captions(chunk: ParsedChunk) -> tuple[str, ...]:
    """
    Extract captions from text or picture metadata.

    Text chunks may carry:
    - metadata["captions"]

    Picture chunks may carry:
    - metadata["caption"]
    """
    values: list[str] = []

    raw_captions = chunk.metadata.get("captions")
    if isinstance(raw_captions, list):
        values.extend(str(item) for item in raw_captions)

    raw_caption = chunk.metadata.get("caption")
    if isinstance(raw_caption, str):
        values.append(raw_caption)

    return _dedupe_keep_order(values)


def _extract_chunk_annotations(chunk: ParsedChunk) -> tuple[str, ...]:
    """Extract annotation strings if present."""
    raw = chunk.metadata.get("annotations")
    if isinstance(raw, list):
        return _dedupe_keep_order(str(item) for item in raw)
    return ()


@dataclass(slots=True)
class _SectionAccumulator:
    """
    Mutable builder-only grouping state for one section path.

    This is intentionally not part of the public domain model.
    """

    doc_id: str
    source_file: str
    heading_path: HeadingPath
    page_start: int | None = None
    page_end: int | None = None
    chunks: list[ParsedChunk] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    visual_clues: list[VisualClue] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)

    def add_chunk(self, chunk: ParsedChunk) -> None:
        """Absorb one parsed chunk into the section group."""
        self.chunks.append(chunk)
        self.chunk_ids.append(chunk.chunk_id)

        self.page_start = _merge_page_start(self.page_start, chunk.page_start)
        self.page_end = _merge_page_end(self.page_end, chunk.page_end)

        captions = _extract_chunk_captions(chunk)
        annotations = _extract_chunk_annotations(chunk)
        block_type = _extract_block_type(chunk)

        self.captions.extend(captions)
        self.annotations.extend(annotations)
        self.block_types.append(block_type)

        if block_type == "picture":
            self.visual_clues.append(
                VisualClue(
                    kind="picture",
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    caption=_clean_text(str(chunk.metadata.get("caption") or "")) or None,
                    annotations=annotations,
                )
            )

        for caption in captions:
            self.visual_clues.append(
                VisualClue(
                    kind="caption",
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    caption=caption,
                    annotations=(),
                )
            )


class MacroPacketBuilder:
    """
    Convert parsed chunks into document-level and section-level macro packets.

    This class does not call the generator and does not persist anything.
    """

    def __init__(
        self,
        *,
        max_section_excerpts: int = 4,
        max_excerpt_chars: int = 1200,
    ) -> None:
        self._max_section_excerpts = max_section_excerpts
        self._max_excerpt_chars = max_excerpt_chars

    def build(self, chunks: Sequence[ParsedChunk]) -> MacroPacketBundle:
        """
        Build one document packet and all section packets for a single document.

        Expectations:
        - all chunks belong to one document
        - chunk order is meaningful and should be preserved
        """
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

        return MacroPacketBundle(
            document=document_packet,
            sections=section_packets,
        )

    def _resolve_single_doc_id(self, chunks: Sequence[ParsedChunk]) -> str:
        """Ensure all chunks belong to one document."""
        doc_ids = {chunk.doc_id for chunk in chunks}
        if len(doc_ids) != 1:
            raise ValueError(
                f"Expected one doc_id in builder input, got: {sorted(doc_ids)}"
            )
        return next(iter(doc_ids))

    def _resolve_single_source_file(self, chunks: Sequence[ParsedChunk]) -> str:
        """Ensure all chunks belong to one source file."""
        source_files = {chunk.source_file for chunk in chunks}
        if len(source_files) != 1:
            raise ValueError(
                f"Expected one source_file in builder input, got: {sorted(source_files)}"
            )
        return next(iter(source_files))

    def _group_sections(
        self,
        chunks: Sequence[ParsedChunk],
    ) -> list[_SectionAccumulator]:
        """
        Group chunks by full heading path in first-seen document order.

        We preserve first-seen order because later document summarization should
        see sections in their original document order.
        """
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

    def _build_section_packet(
        self,
        accumulator: _SectionAccumulator,
    ) -> SectionMacroPacket:
        """Convert one grouped section accumulator into a section packet."""
        excerpts = self._select_representative_excerpts(accumulator.chunks)

        return SectionMacroPacket(
            doc_id=accumulator.doc_id,
            source_file=accumulator.source_file,
            heading_path=accumulator.heading_path,
            page_start=accumulator.page_start,
            page_end=accumulator.page_end,
            chunk_ids=tuple(accumulator.chunk_ids),
            chunk_count=len(accumulator.chunks),
            representative_excerpts=excerpts,
            captions=_dedupe_keep_order(accumulator.captions),
            annotations=_dedupe_keep_order(accumulator.annotations),
            visual_clues=tuple(accumulator.visual_clues),
            block_types=_dedupe_keep_order(accumulator.block_types),
        )

    def _select_representative_excerpts(
        self,
        chunks: Sequence[ParsedChunk],
    ) -> tuple[SectionExcerpt, ...]:
        """
        Pick a few representative excerpts from the section.

        Strategy:
        - prefer non-picture chunks
        - pick first, middle, last
        - add two more positions when the section is larger
        """
        if not chunks:
            return ()

        preferred_chunks = [
            chunk for chunk in chunks if _extract_block_type(chunk) != "picture"
        ]
        source_chunks = preferred_chunks or list(chunks)

        candidate_indices: list[int] = [0]

        if len(source_chunks) >= 3:
            candidate_indices.append(len(source_chunks) // 2)

        if len(source_chunks) >= 2:
            candidate_indices.append(len(source_chunks) - 1)

        if len(source_chunks) > 3 and self._max_section_excerpts > 3:
            candidate_indices.extend(
                [
                    len(source_chunks) // 4,
                    (3 * len(source_chunks)) // 4,
                ]
            )

        excerpts: list[SectionExcerpt] = []
        seen_chunk_ids: set[str] = set()

        for index in candidate_indices:
            chunk = source_chunks[index]
            if chunk.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk.chunk_id)
            excerpts.append(
                SectionExcerpt(
                    chunk_id=chunk.chunk_id,
                    text=self._truncate_excerpt_text(chunk.text),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    rationale=self._excerpt_rationale(index, len(source_chunks)),
                )
            )

            if len(excerpts) >= self._max_section_excerpts:
                break

        return tuple(excerpts)

    def _truncate_excerpt_text(self, text: str) -> str:
        """Keep excerpt text bounded for later summary prompts."""
        normalized = " ".join(text.split()).strip()
        if len(normalized) <= self._max_excerpt_chars:
            return normalized
        return normalized[: self._max_excerpt_chars].rstrip() + "..."

    @staticmethod
    def _excerpt_rationale(index: int, total: int) -> str:
        """Label why a chunk was selected as representative."""
        if index == 0:
            return "section_start"
        if index == total - 1:
            return "section_end"
        if index == total // 2:
            return "section_middle"
        return "section_sample"

    def _build_document_packet(
        self,
        *,
        doc_id: str,
        source_file: str,
        chunks: Sequence[ParsedChunk],
        section_packets: Sequence[SectionMacroPacket],
    ) -> DocumentMacroPacket:
        """Build the document packet from raw chunks and section packets."""
        visual_clues: list[VisualClue] = []
        max_page = 0

        for chunk in chunks:
            if chunk.page_end is not None:
                max_page = max(max_page, chunk.page_end)
            elif chunk.page_start is not None:
                max_page = max(max_page, chunk.page_start)

            if _extract_block_type(chunk) == "picture":
                visual_clues.append(
                    VisualClue(
                        kind="picture",
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        caption=_clean_text(str(chunk.metadata.get("caption") or "")) or None,
                        annotations=_extract_chunk_annotations(chunk),
                    )
                )

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
            visual_clues=tuple(visual_clues),
        )

    def _infer_document_title(
        self,
        *,
        source_file: str,
        section_packets: Sequence[SectionMacroPacket],
    ) -> str:
        """
        Infer a document title cheaply.

        Current priority:
        1. first labeled heading path
        2. filename stem
        """
        for packet in section_packets:
            first_heading = packet.heading_path[0] if packet.heading_path else ""
            if first_heading != UNLABELED_SECTION_TOKEN:
                return " - ".join(packet.heading_path[:2])

        return Path(source_file).stem.replace("_", " ").replace("-", " ").strip()