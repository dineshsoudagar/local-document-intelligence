"""Utilities for building grounded prompt context and rendering answer sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.retrieval.qdrant_hybrid_index import RetrievedChunk
from src.retrieval.qwen_models import LocalQwenGenerator


@dataclass(slots=True)
class AnswerSource:
    """Serializable source reference attached to a generated answer."""

    rank: int
    chunk_id: str
    doc_id: str | None
    source_file: str | None
    original_filename: str | None
    page_start: int | None
    page_end: int | None
    rerank_score: float | None
    fusion_score: float | None
    headings: list[str] | None
    block_type: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "original_filename": self.original_filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "rerank_score": self.rerank_score,
            "fusion_score": self.fusion_score,
            "headings": self.headings,
            "block_type": self.block_type,
        }


@dataclass(slots=True)
class GroundedContext:
    """Grounded prompt context with source references."""

    text: str
    sources: list[AnswerSource]
    used_tokens: int


def build_grounded_context(
    generator: LocalQwenGenerator,
    chunks: Sequence[RetrievedChunk],
    *,
    max_context_tokens: int,
    max_chunk_tokens: int,
) -> GroundedContext:
    """Build a prompt context from retrieved chunks within a token budget."""
    blocks: list[str] = []
    sources: list[AnswerSource] = []
    used_tokens = 0

    for rank, chunk in enumerate(chunks, start=1):
        text = generator.truncate_text(chunk.text.strip(), max_chunk_tokens)
        if not text:
            continue

        headings = chunk.metadata.get("headings")
        heading_text = " > ".join(headings) if isinstance(headings, list) and headings else "-"
        block_type = str(chunk.metadata.get("block_type") or "text")
        source_file = chunk.source_file or "unknown"
        doc_id = chunk.metadata.get("doc_id")
        pages = format_pages(chunk.page_start, chunk.page_end)

        block = (
            f"[{rank}] source_file: {source_file}\n"
            f"[{rank}] pages: {pages}\n"
            f"[{rank}] block_type: {block_type}\n"
            f"[{rank}] headings: {heading_text}\n"
            f"[{rank}] content:\n{text}"
        )
        block_tokens = generator.count_tokens(block)

        if blocks and used_tokens + block_tokens > max_context_tokens:
            break

        if not blocks and block_tokens > max_context_tokens:
            shrink_target = max(128, max_context_tokens // 2)
            trimmed_text = generator.truncate_text(text, shrink_target)
            block = (
                f"[{rank}] source_file: {source_file}\n"
                f"[{rank}] pages: {pages}\n"
                f"[{rank}] block_type: {block_type}\n"
                f"[{rank}] headings: {heading_text}\n"
                f"[{rank}] content:\n{trimmed_text}"
            )
            block_tokens = generator.count_tokens(block)

        if block_tokens > max_context_tokens:
            continue

        blocks.append(block)
        used_tokens += block_tokens
        sources.append(
            AnswerSource(
                rank=rank,
                chunk_id=chunk.chunk_id,
                doc_id=str(doc_id) if doc_id else None,
                source_file=chunk.source_file,
                original_filename=None,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                rerank_score=chunk.rerank_score,
                fusion_score=chunk.fused_score,
                headings=headings,
                block_type=block_type,
            )
        )

    return GroundedContext(
        text="\n\n".join(blocks).strip(),
        sources=sources,
        used_tokens=used_tokens,
    )

def format_pages(page_start: int | None, page_end: int | None) -> str:
    """Return a stable printable page span."""
    if page_start is None and page_end is None:
        return "unknown"
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"


def format_score(value: float | None) -> str:
    """Format a numeric score for display."""
    if value is None:
        return "-"
    return f"{value:.4f}"


def render_sources(sources: Sequence[AnswerSource]) -> str:
    """Render a readable source appendix."""
    lines = ["Sources:"]
    for source in sources:
        heading_text = " > ".join(source.headings) if source.headings else "-"
        lines.append(
            "  "
            f"[{source.rank}] {source.source_file or 'unknown'} | "
            f"pages {format_pages(source.page_start, source.page_end)} | "
            f"chunk_id={source.chunk_id} | "
            f"block_type={source.block_type or '-'} | "
            f"headings={heading_text} | "
            f"rerank={format_score(source.rerank_score)} | "
            f"fusion={format_score(source.fusion_score)}"
        )
    return "\n".join(lines)
