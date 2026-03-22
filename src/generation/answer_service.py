from __future__ import annotations

"""Service layer for grounded answer generation over retrieved chunks."""

import time
from dataclasses import dataclass
from typing import Any, Sequence, Iterator

from src.config.generator_config import GeneratorConfig
from src.generation.context_builder import AnswerSource, GroundedContext, build_grounded_context
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex, RetrievedChunk
from src.retrieval.qwen_models import LocalQwenGenerator
from src.retrieval.query_planner import QueryPlanner  #query-aware retrieval planning


@dataclass(slots=True)
class AnswerTimings:
    """Execution timings for grounded answer generation."""

    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly representation."""
        return {
            "retrieval_seconds": self.retrieval_seconds,
            "generation_seconds": self.generation_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(slots=True)
class GroundedAnswerResult:
    """Structured output returned by the answer service."""

    query: str
    answer: str
    context: GroundedContext
    retrieved_chunk_count: int
    timings: AnswerTimings
    mode_used: str = "grounded"
    fallback_reason: str | None = None

    def to_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        payload: dict[str, Any] = {
            "query": self.query,
            "answer": self.answer,
            "mode_used": self.mode_used,
            "fallback_reason": self.fallback_reason,
            "sources": [source.to_dict() for source in self.context.sources],
            "used_context_tokens": self.context.used_tokens,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "timings": self.timings.to_dict(),
        }
        if include_context:
            payload["context"] = self.context
        return payload


class GroundedAnswerService:
    """Coordinate retrieval, context construction, and grounded generation."""

    def __init__(
        self,
        *,
        index: QdrantHybridIndex,
        config: GeneratorConfig,
        generator: LocalQwenGenerator | None = None,
        planner: QueryPlanner | None = None,  # New: inject planner, allow override in tests
    ) -> None:
        config.validate()
        self._index = index
        self._config = config
        self._generator = generator
        self._planner = planner or QueryPlanner()  # Default planner for normal runtime

    @property
    def generator(self) -> LocalQwenGenerator:
        """Return a lazily initialized local generator."""
        if self._generator is None:
            self._generator = LocalQwenGenerator(self._config.generator_model_path)
        return self._generator

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        """
        Run planned retrieval instead of raw retrieval.

        This is the main behavioral change:
        the planner decides whether this is QA, summary, explanation, or comparison.
        """
        started_at = time.perf_counter()

        chunks = self._planner.retrieve(
            index=self._index,
            query=query,
            doc_ids=doc_ids,
        )

        elapsed = time.perf_counter() - started_at
        return chunks, elapsed

    def build_context(self, chunks: Sequence[RetrievedChunk]) -> GroundedContext:
        """Build prompt context from retrieved chunks."""
        return build_grounded_context(
            self.generator,
            chunks,
            max_context_tokens=self._config.max_context_tokens,
            max_chunk_tokens=self._config.max_chunk_tokens,
        )

    def generate_chat_answer(self, *, query: str) -> tuple[str, float]:
        """Generate a direct non-grounded answer."""
        started_at = time.perf_counter()
        answer = self.generator.generate_chat_answer(
            query=query,
            system_prompt=self._config.assistant_system_prompt,
            chat_instruction=self._config.assistant_instruction,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
        )
        elapsed = time.perf_counter() - started_at
        return answer, elapsed

    def generate_answer(
        self,
        *,
        query: str,
        mode: str,
        context_text: str | None = None,
    ) -> tuple[str, float]:
        """Generate a grounded or direct answer."""
        started_at = time.perf_counter()

        if mode == "grounded":
            system_prompt = self._config.system_prompt
            answer_instruction = self._config.answer_instruction
        elif mode == "chat":
            system_prompt = self._config.assistant_system_prompt
            answer_instruction = self._config.assistant_instruction
        else:
            raise ValueError(f"Unsupported generation mode: {mode}")

        answer = self.generator.generate_answer(
            query=query,
            context=context_text,
            system_prompt=system_prompt,
            answer_instruction=answer_instruction,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
        )
        elapsed = time.perf_counter() - started_at
        return answer, elapsed

    def stream_answer(
        self,
        query: str,
        doc_ids: list[str] | None = None,
    ) -> tuple[GroundedContext, int, float, Iterator[str]]:
        """Prepare grounded context and return a text stream for generation."""
        retrieved_chunks, retrieval_seconds = self.retrieve(query, doc_ids=doc_ids)

        if not retrieved_chunks:
            def empty_stream() -> Iterator[str]:
                yield "No relevant chunks were retrieved."

            return (
                GroundedContext(text="", sources=[], used_tokens=0),
                0,
                retrieval_seconds,
                empty_stream(),
            )

        grounded_context = self.build_context(retrieved_chunks)
        if not grounded_context.text:
            raise RuntimeError(
                "Retrieved chunks were available, but no context could fit the prompt budget"
            )

        prompt = self.generator.build_prompt(
            query=query,
            context=grounded_context.text,
            system_prompt=self._config.system_prompt,
            answer_instruction=self._config.answer_instruction,
        )
        stream = self.generator.stream_from_prompt(
            prompt,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
        )
        return grounded_context, len(retrieved_chunks), retrieval_seconds, stream