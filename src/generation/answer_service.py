"""Service layer for grounded and direct answer generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from src.config.generator_config import GeneratorConfig
from src.generation.context_builder import (
    AnswerSource,
    GroundedContext,
    build_grounded_context,
)
from src.config.retrieval_control_config import (
    RetrievalControlConfig,
)
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex, RetrievedChunk
from src.retrieval.qwen_models import GeneratedText, LocalQwenGenerator, StreamEvent
from src.retrieval.controller_service import AutoDecisionController

@dataclass(slots=True)
class AnswerTimings:
    """Execution timings for one query."""

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
class StreamStartPayload:
    """Initial metadata sent before streamed answer text."""

    query: str
    mode_used: str
    reasoning_mode: str
    fallback_reason: str | None
    sources: list[AnswerSource]
    used_context_tokens: int
    retrieved_chunk_count: int
    retrieval_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "query": self.query,
            "mode_used": self.mode_used,
            "reasoning_mode": self.reasoning_mode,
            "fallback_reason": self.fallback_reason,
            "sources": [source.to_dict() for source in self.sources],
            "used_context_tokens": self.used_context_tokens,
            "retrieved_chunk_count": self.retrieved_chunk_count,
            "retrieval_seconds": self.retrieval_seconds,
        }


@dataclass(slots=True)
class GroundedAnswerResult:
    """Structured answer returned by the service."""

    query: str
    answer: str
    context: GroundedContext
    retrieved_chunk_count: int
    timings: AnswerTimings
    mode_used: str
    reasoning_mode: str
    thinking_content: str | None = None
    thinking_finished: bool = False
    fallback_reason: str | None = None

    def to_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        payload: dict[str, Any] = {
            "query": self.query,
            "answer": self.answer,
            "mode_used": self.mode_used,
            "reasoning_mode": self.reasoning_mode,
            "thinking_content": self.thinking_content,
            "thinking_finished": self.thinking_finished,
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
    """Coordinate retrieval, context construction, and answer generation."""

    def __init__(self, *, index: QdrantHybridIndex, config: GeneratorConfig,
                 generator: LocalQwenGenerator | None = None, ) -> None:
        config.validate()
        self._index = index
        self._config = config
        self._generator = generator
        self._retrieval_config = RetrievalControlConfig()
        self._controller = AutoDecisionController(
            generator=self.generator,
            config=self._retrieval_config.auto,
        )
    @property
    def generator(self) -> LocalQwenGenerator:
        """Return a lazily initialized local generator."""
        if self._generator is None:
            self._generator = LocalQwenGenerator.from_config(self._config)
        return self._generator

    def answer(
        self,
        query: str,
        *,
        mode: str = "grounded",
        doc_ids: list[str] | None = None,
        reasoning_mode: str = "no_think",
    ) -> GroundedAnswerResult:
        """Return a buffered answer for grounded, chat, or auto mode."""
        self._validate_mode(mode)
        resolved_reasoning_mode = self._config.resolve_reasoning_mode(reasoning_mode)
        overall_started_at = time.perf_counter()

        if mode == "chat":
            generated, generation_seconds = self._generate_text(
                query=query,
                mode="chat",
                context_text=None,
                reasoning_mode=resolved_reasoning_mode,
            )
            return self._build_result(
                query=query,
                generated=generated,
                context=self._empty_context(),
                retrieved_chunk_count=0,
                retrieval_seconds=0.0,
                generation_seconds=generation_seconds,
                overall_started_at=overall_started_at,
                mode_used="chat",
                reasoning_mode=resolved_reasoning_mode,
            )

        retrieved_chunks, retrieval_seconds = self._retrieve_chunks(
            query,
            doc_ids=doc_ids,
        )
        resolved_mode, fallback_reason = self._resolve_mode(mode, retrieved_chunks)

        if resolved_mode == "chat":
            generated, generation_seconds = self._generate_text(
                query=query,
                mode="chat",
                context_text=None,
                reasoning_mode=resolved_reasoning_mode,
            )
            return self._build_result(
                query=query,
                generated=generated,
                context=self._empty_context(),
                retrieved_chunk_count=len(retrieved_chunks),
                retrieval_seconds=retrieval_seconds,
                generation_seconds=generation_seconds,
                overall_started_at=overall_started_at,
                mode_used="chat",
                reasoning_mode=resolved_reasoning_mode,
                fallback_reason=fallback_reason,
            )

        if not retrieved_chunks:
            return self._build_result(
                query=query,
                generated=GeneratedText(
                    answer="No relevant chunks were retrieved.",
                    thinking_content=None,
                    thinking_finished=(resolved_reasoning_mode == "no_think"),
                ),
                context=self._empty_context(),
                retrieved_chunk_count=0,
                retrieval_seconds=retrieval_seconds,
                generation_seconds=0.0,
                overall_started_at=overall_started_at,
                mode_used="grounded",
                reasoning_mode=resolved_reasoning_mode,
            )

        grounded_context = self._build_context(retrieved_chunks)
        if not grounded_context.text:
            raise RuntimeError(
                "Retrieved chunks were available, but no context could fit the prompt budget"
            )

        generated, generation_seconds = self._generate_text(
            query=query,
            mode="grounded",
            context_text=grounded_context.text,
            reasoning_mode=resolved_reasoning_mode,
        )
        return self._build_result(
            query=query,
            generated=generated,
            context=grounded_context,
            retrieved_chunk_count=len(retrieved_chunks),
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            overall_started_at=overall_started_at,
            mode_used="grounded",
            reasoning_mode=resolved_reasoning_mode,
        )

    def stream(
        self,
        query: str,
        *,
        mode: str = "grounded",
        doc_ids: list[str] | None = None,
        reasoning_mode: str = "no_think",
        stream_thinking: bool = False,
    ) -> tuple[StreamStartPayload, Iterator[StreamEvent]]:
        """Return response metadata and a streamed answer iterator."""
        self._validate_mode(mode)
        resolved_reasoning_mode = self._config.resolve_reasoning_mode(reasoning_mode)

        if mode == "chat":
            return self._build_stream_response(
                query=query,
                mode_used="chat",
                reasoning_mode=resolved_reasoning_mode,
                context=self._empty_context(),
                retrieved_chunk_count=0,
                retrieval_seconds=0.0,
                fallback_reason=None,
                stream_thinking=stream_thinking,
            )

        if doc_ids is None and mode == "auto":
            decision = self._controller.decide(query)
            auto_decision = decision.decision
            reason = decision.reason_short
            if auto_decision == "chat":
                return self._build_stream_response(
                    query=query,
                    mode_used="chat",
                    reasoning_mode=resolved_reasoning_mode,
                    context=self._empty_context(),
                    retrieved_chunk_count=0,
                    retrieval_seconds=0.0,
                    fallback_reason=reason,
                    stream_thinking=stream_thinking,
                )

        retrieved_chunks, retrieval_seconds = self._retrieve_chunks(
            query,
            doc_ids=doc_ids,
        )

        if not retrieved_chunks:
            start_payload = StreamStartPayload(
                query=query,
                mode_used="grounded",
                reasoning_mode=resolved_reasoning_mode,
                fallback_reason=None,
                sources=[],
                used_context_tokens=0,
                retrieved_chunk_count=0,
                retrieval_seconds=retrieval_seconds,
            )
            return start_payload, self._empty_stream_events()

        grounded_context = self._build_context(retrieved_chunks)
        if not grounded_context.text:
            raise RuntimeError(
                "Retrieved chunks were available, but no context could fit the prompt budget"
            )

        return self._build_stream_response(
            query=query,
            mode_used="grounded",
            reasoning_mode=resolved_reasoning_mode,
            context=grounded_context,
            retrieved_chunk_count=len(retrieved_chunks),
            retrieval_seconds=retrieval_seconds,
            fallback_reason=None,
            stream_thinking=stream_thinking,
        )

    def _retrieve_chunks(self, query: str, *,
                         doc_ids: list[str] | None = None) -> tuple[list[RetrievedChunk], float]:
        """Run retrieval for the supplied query."""
        started_at = time.perf_counter()
        retrieved_chunks = self._index.search(query, doc_ids=doc_ids)
        retrieval_seconds = time.perf_counter() - started_at
        return retrieved_chunks, retrieval_seconds

    def _build_context(
            self,
            retrieved_chunks: Sequence[RetrievedChunk],
    ) -> GroundedContext:
        """Build prompt context from retrieved evidence."""
        return build_grounded_context(
            self.generator,
            retrieved_chunks,
            max_context_tokens=self._config.max_context_tokens,
            max_chunk_tokens=self._config.max_chunk_tokens,
        )

    def _generate_text(
        self,
        *,
        query: str,
        mode: str,
        context_text: str | None,
        reasoning_mode: str,
    ) -> tuple[GeneratedText, float]:
        system_prompt, answer_instruction = self._prompts_for_mode(mode)
        enable_thinking = self._config.thinking_enabled(reasoning_mode)

        started_at = time.perf_counter()
        result = self.generator.generate_answer(
            query=query,
            context=context_text,
            system_prompt=system_prompt,
            answer_instruction=answer_instruction,
            max_new_tokens=self._config.max_new_tokens_for(reasoning_mode),
            temperature=self._config.temperature_for(reasoning_mode),
            top_p=self._config.top_p_for(reasoning_mode),
            repetition_penalty=self._config.repetition_penalty,
            enable_thinking=enable_thinking,
            return_thinking=enable_thinking,
        )
        generation_seconds = time.perf_counter() - started_at
        return result, generation_seconds

    def _stream_text(
            self,
            *,
            query: str,
            mode: str,
            context_text: str | None,
            reasoning_mode: str,
            stream_thinking: bool,
    ) -> Iterator[StreamEvent]:
        """Stream answer events incrementally."""
        system_prompt, answer_instruction = self._prompts_for_mode(mode)
        enable_thinking = self._config.thinking_enabled(reasoning_mode)

        prompt = self.generator.build_prompt(
            query=query,
            context=context_text,
            system_prompt=system_prompt,
            answer_instruction=answer_instruction,
            enable_thinking=enable_thinking,
        )
        return self.generator.stream_from_prompt(
            prompt,
            max_new_tokens=self._config.max_new_tokens_for(reasoning_mode),
            temperature=self._config.temperature_for(reasoning_mode),
            top_p=self._config.top_p_for(reasoning_mode),
            repetition_penalty=self._config.repetition_penalty,
            enable_thinking=enable_thinking,
            stream_thinking=stream_thinking,
        )

    def _resolve_mode(
            self,
            requested_mode: str,
            retrieved_chunks: Sequence[RetrievedChunk],
    ) -> tuple[str, str | None]:
        """Resolve auto mode into grounded or chat."""
        if requested_mode == "grounded":
            return "grounded", None
        if requested_mode == "chat":
            return "chat", None
        if requested_mode == "auto":
            use_grounded, fallback_reason = self._should_use_grounded(retrieved_chunks)
            if use_grounded:
                return "grounded", None
            return "chat", fallback_reason
        raise ValueError(f"Unsupported mode: {requested_mode}")

    def _prompts_for_mode(self, mode: str) -> tuple[str, str]:
        """Return prompt settings for grounded or chat generation."""
        if mode == "grounded":
            return self._config.system_prompt, self._config.answer_instruction
        if mode == "chat":
            return (
                self._config.assistant_system_prompt,
                self._config.assistant_instruction,
            )
        raise ValueError(f"Unsupported generation mode: {mode}")

    def _build_result(
            self,
            *,
            query: str,
            generated: GeneratedText,
            context: GroundedContext,
            retrieved_chunk_count: int,
            retrieval_seconds: float,
            generation_seconds: float,
            overall_started_at: float,
            mode_used: str,
            reasoning_mode: str,
            fallback_reason: str | None = None,
    ) -> GroundedAnswerResult:
        """Create a structured query result with timings."""
        timings = AnswerTimings(
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - overall_started_at,
        )
        return GroundedAnswerResult(
            query=query,
            answer=generated.answer,
            context=context,
            retrieved_chunk_count=retrieved_chunk_count,
            timings=timings,
            mode_used=mode_used,
            reasoning_mode=reasoning_mode,
            thinking_content=generated.thinking_content,
            thinking_finished=generated.thinking_finished,
            fallback_reason=fallback_reason,
        )

    def _build_stream_response(
            self,
            *,
            query: str,
            mode_used: str,
            reasoning_mode: str,
            context: GroundedContext,
            retrieved_chunk_count: int,
            retrieval_seconds: float,
            fallback_reason: str | None,
            stream_thinking: bool,
    ) -> tuple[StreamStartPayload, Iterator[StreamEvent]]:
        """Build stream metadata and the corresponding event iterator."""
        start_payload = StreamStartPayload(
            query=query,
            mode_used=mode_used,
            reasoning_mode=reasoning_mode,
            fallback_reason=fallback_reason,
            sources=context.sources,
            used_context_tokens=context.used_tokens,
            retrieved_chunk_count=retrieved_chunk_count,
            retrieval_seconds=retrieval_seconds,
        )
        event_stream = self._stream_text(
            query=query,
            mode=mode_used,
            context_text=context.text or None,
            reasoning_mode=reasoning_mode,
            stream_thinking=stream_thinking,
        )
        return start_payload, event_stream

    def _should_use_grounded(
            self,
            retrieved_chunks: Sequence[RetrievedChunk],
    ) -> tuple[bool, str | None]:
        """Decide whether retrieval is strong enough for grounded answering."""
        if not retrieved_chunks:
            return False, "no_retrieval_hits"

        top_score = retrieved_chunks[0].rerank_score
        if top_score is None:
            return False, "missing_rerank_score"

        if top_score >= self._config.auto_min_top_rerank_score:
            return True, None

        return False, f"low_top_rerank_score: {top_score:.4f}"

    @staticmethod
    def _empty_context() -> GroundedContext:
        """Return an empty context for non-grounded responses."""
        return GroundedContext(text="", sources=[], used_tokens=0)

    @staticmethod
    def _empty_stream_events() -> Iterator[StreamEvent]:
        """Return a one-message event stream for empty retrieval results."""
        yield StreamEvent(kind="answer_token", text="No relevant chunks were retrieved.")

    @staticmethod
    def _validate_mode(mode: str) -> None:
        """Validate the public query mode."""
        if mode not in {"grounded", "chat", "auto"}:
            raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def sources_to_dicts(sources: Sequence[AnswerSource]) -> list[dict[str, Any]]:
        """Convert answer sources into JSON-friendly dictionaries."""
        return [source.to_dict() for source in sources]
