"""Generate section-level and document-level macro summaries from macro packets."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.generator_config import GeneratorConfig
from src.indexing.macro_profiles import (
    DocumentMacroPacket,
    DocumentMacroProfile,
    MacroSummaryBundle,
    SectionMacroPacket,
    SectionMacroProfile,
)
from src.retrieval.qwen_models import LocalQwenGenerator


SECTION_SYSTEM_PROMPT = """You are a document analysis assistant.
Return only valid JSON.
Write grounded summaries from the supplied section text.
Do not invent facts not present in the section text."""

DOCUMENT_SYSTEM_PROMPT = """You are a document analysis assistant.
Return only valid JSON.
Write grounded summaries from the supplied section summaries.
Do not invent facts not present in the inputs."""


@dataclass(slots=True)
class MacroSummaryService:
    """Create minimal section and document summaries from macro packets."""

    generator: LocalQwenGenerator
    config: GeneratorConfig

    @classmethod
    def from_config(cls, config: GeneratorConfig | None = None) -> "MacroSummaryService":
        resolved_config = config or GeneratorConfig()
        return cls(
            generator=LocalQwenGenerator.from_config(resolved_config),
            config=resolved_config,
        )

    def summarize_document(self, packet: DocumentMacroPacket) -> MacroSummaryBundle:
        """Summarize all sections, then summarize the full document."""
        section_profiles = tuple(
            self.summarize_section(section_packet)
            for section_packet in packet.section_packets
        )

        document_profile = self._summarize_document_profile(
            packet=packet,
            section_profiles=section_profiles,
        )

        return MacroSummaryBundle(
            document=document_profile,
            sections=section_profiles,
        )

    def summarize_section(self, packet: SectionMacroPacket) -> SectionMacroProfile:
        """Summarize one section packet."""
        section_text = packet.section_text.strip()
        if not section_text:
            summary = ""
            keywords: tuple[str, ...] = ()
        else:
            if self._needs_split(section_text):
                summary, keywords = self._summarize_large_section(packet)
            else:
                summary, keywords = self._summarize_section_text(
                    section_heading=packet.section_heading,
                    section_text=section_text,
                )

        return SectionMacroProfile(
            section_id=packet.section_id,
            doc_id=packet.doc_id,
            source_file=packet.source_file,
            heading_path=packet.heading_path,
            section_heading=packet.section_heading,
            page_start=packet.page_start,
            page_end=packet.page_end,
            section_summary=summary,
            keywords=keywords,
        )

    def _summarize_large_section(
        self,
        packet: SectionMacroPacket,
    ) -> tuple[str, tuple[str, ...]]:
        """Summarize oversized section text by recursive chunk summarization."""
        parts = self._split_text_by_tokens(packet.section_text)
        part_summaries: list[str] = []

        for part in parts:
            part_summary, _ = self._summarize_section_text(
                section_heading=packet.section_heading,
                section_text=part,
            )
            if part_summary:
                part_summaries.append(part_summary)

        merged_text = "\n\n".join(part_summaries).strip()
        return self._summarize_section_text(
            section_heading=packet.section_heading,
            section_text=merged_text,
        )

    def _summarize_section_text(
        self,
        *,
        section_heading: str,
        section_text: str,
    ) -> tuple[str, tuple[str, ...]]:
        """Run one structured generation call for a section."""
        user_prompt = (
            "Summarize the following document section.\n\n"
            f"Section heading: {section_heading}\n\n"
            "Return JSON with exactly these keys:\n"
            '{\n'
            '  "section_summary": "2-4 sentence grounded summary",\n'
            '  "keywords": ["keyword 1", "keyword 2", "keyword 3", "keyword 4", "keyword 5"]\n'
            '}\n\n'
            f"Section text:\n{section_text}"
        )

        result = self.generator.generate_structured_json(
            system_prompt=SECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=min(512, self.config.max_new_tokens),
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            enable_thinking=False,
        )

        summary = str(result.get("section_summary", "")).strip()
        keywords_raw = result.get("keywords", [])
        keywords = self._normalize_keywords(keywords_raw)
        return summary, keywords

    def _summarize_document_profile(
        self,
        *,
        packet: DocumentMacroPacket,
        section_profiles: tuple[SectionMacroProfile, ...],
    ) -> DocumentMacroProfile:
        """Summarize the full document from section summaries."""
        section_lines: list[str] = []

        for section in section_profiles:
            section_lines.append(
                f"Section: {section.section_heading}\n"
                f"Summary: {section.section_summary}\n"
                f"Keywords: {', '.join(section.keywords)}"
            )

        user_prompt = (
            "Summarize the following document from its section summaries.\n\n"
            f"Document title: {packet.title or ''}\n\n"
            "Return JSON with exactly these keys:\n"
            '{\n'
            '  "doc_summary": "3-6 sentence grounded summary",\n'
            '  "keywords": ["keyword 1", "keyword 2", "keyword 3", "keyword 4", "keyword 5"]\n'
            '}\n\n'
            "Section summaries:\n"
            f"{chr(10).join(section_lines)}"
        )

        result = self.generator.generate_structured_json(
            system_prompt=DOCUMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=min(640, self.config.max_new_tokens),
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            enable_thinking=False,
        )

        return DocumentMacroProfile(
            doc_id=packet.doc_id,
            source_file=packet.source_file,
            title=packet.title,
            page_count=packet.page_count,
            chunk_count=packet.chunk_count,
            doc_summary=str(result.get("doc_summary", "")).strip(),
            keywords=self._normalize_keywords(result.get("keywords", [])),
        )

    def _needs_split(self, text: str) -> bool:
        """Return whether the section text should be recursively summarized."""
        token_count = self.generator.count_tokens(text)
        return token_count > int(self.config.max_context_tokens * 0.7)

    def _split_text_by_tokens(self, text: str) -> list[str]:
        """Split text into token-safe parts."""
        words = text.split()
        if not words:
            return [""]

        parts: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate_words = current_words + [word]
            candidate_text = " ".join(candidate_words)

            if self.generator.count_tokens(candidate_text) <= int(self.config.max_context_tokens * 0.5):
                current_words = candidate_words
                continue

            if current_words:
                parts.append(" ".join(current_words))
                current_words = [word]
            else:
                parts.append(word)
                current_words = []

        if current_words:
            parts.append(" ".join(current_words))

        return parts

    def _normalize_keywords(self, value: object) -> tuple[str, ...]:
        """Normalize keyword output from the model."""
        if not isinstance(value, list):
            return ()

        result: list[str] = []
        seen: set[str] = set()

        for item in value:
            keyword = str(item).strip()
            if not keyword:
                continue

            key = keyword.casefold()
            if key in seen:
                continue

            seen.add(key)
            result.append(keyword)

        return tuple(result[:10])