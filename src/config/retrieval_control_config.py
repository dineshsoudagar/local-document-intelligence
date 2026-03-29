"""Configuration types for retrieval planning, judging, and answer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RetrievalMode = Literal["auto", "chat", "corpus", "selected_document", "grounded"]
EvidenceVerdict = Literal["supported_answer", "partial_support", "unsupported"]
AutoDecisionLabel = Literal["chat", "retrieve"]


@dataclass(frozen=True, slots=True)
class AutoControllerConfig:
    """Configuration for the auto front-door chat vs retrieve gate."""

    min_chat_confidence: float = 0.72
    max_new_tokens: int = 128
    max_retries: int = 1
    system_prompt: str = (
        "You are a retrieval gatekeeper for a document intelligence system. "
        "Decide whether the user query should be answered from general chat knowledge "
        "or should trigger document retrieval. "
        "Return only valid JSON. Do not add explanation before or after the JSON."
    )
    user_instruction: str = (
        "Choose retrieve when the user asks about documents, files, papers, contracts, "
        "reports, notes, uploaded materials, or asks for summaries, explanations, "
        "comparisons, critiques, or extractions grounded in documents. "
        "Choose chat when the request is general knowledge, greeting, casual conversation, "
        "brainstorming, writing help, or system meta discussion unrelated to stored documents. "
        "When uncertain, prefer retrieve if the request could reasonably depend on documents.\n\n"
        "Confidence must be a float between 0.0 and 1.0.\n"
        "Examples:\n"
        "- 1.0 means fully confident\n"
        "- 0.5 means moderately confident\n"
        "- 0.0 means no confidence\n\n"
        "Return exactly this JSON schema:\n"
        '{'
        '"decision":"chat" or "retrieve",'
        '"confidence":0.0,'
        '"reason_short":"short_snake_case_phrase"'
        '}\n\n'
        "Do not use percentages. Do not use strings for confidence. "
        "Do not return markdown. Do not return prose."
    )


@dataclass(frozen=True, slots=True)
class RewriteConfig:
    """Configuration for retrieval-oriented query expansion."""

    min_rewrites: int = 10
    max_rewrites: int = 10
    max_new_tokens: int = 256
    temperature: float = 0.50
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    reasoning_mode: str = "no_think"
    system_prompt: str = (
        "You generate retrieval probes for a document search system. "
        "Do not answer the question. Return only valid JSON."
    )
    user_instruction: str = (
        "Include the original query as one rewrite. Generate concise retrieval rewrites with real semantic spread. "
        "Most rewrites should add a new angle using synonyms, likely section names, domain terminology, related concepts, "
        "abbreviations, constraints, or evidence-focused phrasing. Prefer document-language and search-language over natural "
        "conversational rewording. Do not fill the list with simple paraphrases or reordered wording. Avoid duplicates, "
        "near-duplicates, and unrelated expansions.\n\n"
        "Return JSON with fields: rewrites, keywords, entities."
    )


@dataclass(frozen=True, slots=True)
class FocusedSecondPassConfig:
    """Configuration for the bounded corrective retrieval pass."""

    enabled: bool = False
    retry_top_rerank_below: float = 0.40
    retry_support_rerank_threshold: float = 0.32
    retry_min_supporting_chunks: int = 2

    top_k_per_rewrite: int = 10
    final_top_k: int = 10

    rerank_keep_threshold: float = 0.34
    max_chunks_per_heading: int = 2
    max_chunks_per_section: int = 1
    max_chunks_per_document: int = 4

    chunk_profiles: tuple[str, ...] = ("large",)
    rerank_instruction: str = (
        "Judge whether the document chunk adds missing context or broader evidence needed "
        "to answer the user query."
    )


@dataclass(frozen=True, slots=True)
class RetrievalPassConfig:
    """Configuration for the main retrieval pass."""

    top_k_per_rewrite: int = 16
    global_top_k: int = 64
    final_top_k: int = 14
    max_chunks_per_heading: int = 2
    max_chunks_per_page: int = 2
    chunk_profiles: tuple[str, ...] = ("standard",)
    rerank_instruction: str = (
        "Judge whether the document chunk contains evidence that helps answer the user query faithfully."
    )


@dataclass(frozen=True, slots=True)
class EvidenceJudgeConfig:
    """Thresholds and prompts for evidence sufficiency judgment."""

    strong_top_rerank: float = 0.55
    medium_top_rerank: float = 0.40
    weak_top_rerank: float = 0.22
    support_second_rerank: float = 0.34
    support_third_rerank: float = 0.28
    max_chunk_chars: int = 900
    max_chunks_for_model: int = 5
    max_new_tokens: int = 160
    system_prompt: str = (
        "You are checking whether retrieved document chunks contain enough evidence to answer "
        "a user query. You are not answering the query unless the evidence supports it. "
        "Return only valid JSON."
    )
    user_instruction: str = (
        "Given the user query and top retrieved chunks, decide whether the evidence is "
        "supported_answer, partial_support, or unsupported. Do not assume facts not present "
        "in the chunks. Prefer unsupported over guessing. Choose partial_support when the "
        "evidence is related but incomplete. Choose should_retry true only if another retrieval "
        "pass could plausibly help.\n\n"
        "Return JSON with fields: verdict, confidence, rationale_short, should_retry."
    )


@dataclass(frozen=True, slots=True)
class ResponseStyleConfig:
    """Prompts for grounded, partial, and chat responses."""

    supported_system_prompt: str = (
        "You answer questions only from the provided context. "
        "If the context is insufficient, say the answer is not supported. "
        "Cite supporting evidence inline with bracketed source ids like [1] or [2]. "
        "Do not invent facts, page numbers, or source ids. "
        "Return only the final answer."
    )
    supported_instruction: str = (
        "Write a concise grounded answer using only supported evidence."
    )

    partial_system_prompt: str = (
        "You answer only the portion of the question supported by the provided context. "
        "State clearly that the evidence is incomplete. Cite supporting evidence inline "
        "with bracketed source ids. End with exactly one focused follow-up question. "
        "Return only the final answer."
    )
    partial_instruction: str = (
        "Answer only what is supported. Explicitly state what remains unsupported. "
        "Ask exactly one focused follow-up question."
    )

    chat_system_prompt: str = (
        "You are a helpful local assistant. Answer clearly and directly. "
        "Be honest about uncertainty. Return only the final answer."
    )
    chat_instruction: str = "Answer the user's request directly."


@dataclass(frozen=True, slots=True)
class MultiChunkConfig:
    """Chunk-size profiles used at ingestion and retrieval time."""

    enabled: bool = True
    standard_max_chunk_tokens: int = 260
    standard_min_chunk_tokens: int = 80
    large_max_chunk_tokens: int = 520
    large_min_chunk_tokens: int = 160


@dataclass(slots=True)
class RetrievalControlConfig:
    """Top-level retrieval control configuration."""

    auto: AutoControllerConfig = field(default_factory=AutoControllerConfig)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    pass1: RetrievalPassConfig = field(default_factory=RetrievalPassConfig)
    pass2: FocusedSecondPassConfig = field(default_factory=FocusedSecondPassConfig)
    evidence: EvidenceJudgeConfig = field(default_factory=EvidenceJudgeConfig)
    response: ResponseStyleConfig = field(default_factory=ResponseStyleConfig)
    multi_chunk: MultiChunkConfig = field(default_factory=MultiChunkConfig)
