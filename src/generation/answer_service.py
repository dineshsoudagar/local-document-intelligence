from __future__ import annotations

"""Service layer for grounded answer generation over retrieved chunks."""

import time
from dataclasses import dataclass
from typing import Any, Sequence

from src.config.generator_config import GeneratorConfig
from src.generation.context_builder import AnswerSource, GroundedContext, build_grounded_context
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex, RetrievedChunk
from src.retrieval.qwen_models import LocalQwenGenerator


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

    def to_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        payload: dict[str, Any] = {
            "query": self.query,
            "answer": self.answer,
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
    ) -> None:
        config.validate()
        self._index = index
        self._config = config
        self._generator = generator

    @property
    def generator(self) -> LocalQwenGenerator:
        """Return a lazily initialized local generator."""
        if self._generator is None:
            self._generator = LocalQwenGenerator(self._config.generator_model_path)
        return self._generator

    def retrieve(self, query: str) -> tuple[list[RetrievedChunk], float]:
        """Run retrieval for the supplied query."""
        started_at = time.perf_counter()
        chunks = self._index.search(query, top_k=self._config.retrieval_top_k)
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

    def generate_answer(self, *, query: str, context_text: str) -> tuple[str, float]:
        """Generate a grounded answer from prepared context."""
        started_at = time.perf_counter()
        answer = self.generator.generate_grounded_answer(
            query=query,
            context=context_text,
            system_prompt=self._config.system_prompt,
            answer_instruction=self._config.answer_instruction,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            repetition_penalty=self._config.repetition_penalty,
        )
        elapsed = time.perf_counter() - started_at
        return answer, elapsed

    def answer(self, query: str) -> GroundedAnswerResult:
        """Retrieve evidence and generate a grounded answer."""
        overall_started_at = time.perf_counter()
        retrieved_chunks, retrieval_seconds = self.retrieve(query)

        if not retrieved_chunks:
            timings = AnswerTimings(
                retrieval_seconds=retrieval_seconds,
                generation_seconds=0.0,
                total_seconds=time.perf_counter() - overall_started_at,
            )
            return GroundedAnswerResult(
                query=query,
                answer="No relevant chunks were retrieved.",
                context=GroundedContext(text="", sources=[], used_tokens=0),
                retrieved_chunk_count=0,
                timings=timings,
            )

        grounded_context = self.build_context(retrieved_chunks)
        if not grounded_context.text:
            raise RuntimeError(
                "Retrieved chunks were available, but no context could fit the prompt budget"
            )

        answer, generation_seconds = self.generate_answer(
            query=query,
            context_text=grounded_context.text,
        )
        timings = AnswerTimings(
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - overall_started_at,
        )
        return GroundedAnswerResult(
            query=query,
            answer=answer,
            context=grounded_context,
            retrieved_chunk_count=len(retrieved_chunks),
            timings=timings,
        )

    def answer_from_chunks(
            self,
            *,
            query: str,
            chunks: Sequence[RetrievedChunk],
    ) -> GroundedAnswerResult:
        """Generate a grounded answer from pre-retrieved chunks."""
        overall_started_at = time.perf_counter()
        grounded_context = self.build_context(chunks)
        if not grounded_context.text:
            raise RuntimeError(
                "Provided chunks did not produce any prompt context within the token budget"
            )
        answer, generation_seconds = self.generate_answer(
            query=query,
            context_text=grounded_context.text,
        )
        timings = AnswerTimings(
            retrieval_seconds=0.0,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - overall_started_at,
        )
        return GroundedAnswerResult(
            query=query,
            answer=answer,
            context=grounded_context,
            retrieved_chunk_count=len(chunks),
            timings=timings,
        )

    @staticmethod
    def sources_to_dicts(sources: Sequence[AnswerSource]) -> list[dict[str, Any]]:
        """Convert answer sources into JSON-friendly dictionaries."""
        return [source.to_dict() for source in sources]
