from __future__ import annotations

"""Plan retrieval strategy from the user query and post-process retrieved chunks."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex, RetrievedChunk


# Query intent tells us what kind of retrieval policy to use.
QueryIntent = Literal["factoid", "broad_summary", "explanation", "comparison"]

# Query scope tells us whether the user is asking about one selected document
# or about the whole corpus.
QueryScope = Literal["document", "corpus"]


@dataclass(slots=True)
class RetrievalPlan:
    """Planned retrieval policy for one user query."""

    query: str
    intent: QueryIntent
    scope: QueryScope

    # One user query may expand into multiple retrieval probes.
    retrieval_queries: list[str]

    # The reranker should score chunks differently for QA vs summary.
    rerank_instruction: str

    # candidate_top_k:
    # how many chunks to retrieve per query before merging
    candidate_top_k: int

    # final_top_k:
    # how many chunks survive after merging, rescoring, and diversity filtering
    final_top_k: int

    # Diversity controls to avoid many chunks from the same section/page.
    max_chunks_per_heading: int
    max_chunks_per_page: int

    # Section priors for summary-style requests.
    preferred_heading_terms: tuple[str, ...] = ()
    discouraged_heading_terms: tuple[str, ...] = ()


class QueryPlanner:
    """Decide retrieval strategy and assemble final chunk candidates."""

    # Signals that the query is broad and asks for document-level understanding.
    _summary_cues = (
        "summarize",
        "summary",
        "overview",
        "main idea",
        "main takeaway",
        "key takeaway",
        "what is this document about",
        "what is this paper about",
        "what is this about",
        "tl;dr",
        "gist",
    )

    # Signals that the user wants comparison rather than direct QA.
    _comparison_cues = (
        "compare",
        "comparison",
        "difference",
        "differences",
        "similarities",
        "similarity",
        "versus",
        "vs",
    )

    # Signals that the user wants explanation rather than a short fact.
    _explanation_cues = (
        "explain",
        "walk me through",
        "how does",
        "how do",
        "why does",
        "why is",
        "help me understand",
    )

    def plan(self, *, query: str, doc_ids: list[str] | None) -> RetrievalPlan:
        """
        Classify the query and return a retrieval plan.

        This is the key step that your current pipeline is missing.
        """
        normalized = self._normalize(query)

        # If exactly one document is selected, treat the scope as document-level.
        scope: QueryScope = "document" if len(doc_ids or []) == 1 else "corpus"

        if self._contains_any(normalized, self._comparison_cues):
            return self._build_comparison_plan(query=query, scope=scope)

        if self._contains_any(normalized, self._summary_cues):
            return self._build_summary_plan(query=query, scope=scope)

        if self._contains_any(normalized, self._explanation_cues):
            return self._build_explanation_plan(query=query, scope=scope)

        return self._build_factoid_plan(query=query, scope=scope)

    def retrieve(
        self,
        *,
        index: QdrantHybridIndex,
        query: str,
        doc_ids: list[str] | None,
    ) -> list[RetrievedChunk]:
        """
        Execute the retrieval plan end to end.

        Flow:
        1. classify query
        2. run one or more searches
        3. merge duplicate chunk hits
        4. apply section-aware bonuses/penalties
        5. keep diverse chunks
        """
        plan = self.plan(query=query, doc_ids=doc_ids)

        batches: list[list[RetrievedChunk]] = []

        # Run retrieval once for each planned retrieval query.
        for retrieval_query in plan.retrieval_queries:
            batches.append(
                self._search(
                    index=index,
                    query=retrieval_query,
                    doc_ids=doc_ids,
                    top_k=plan.candidate_top_k,
                    rerank_instruction=plan.rerank_instruction,
                )
            )

        # Merge duplicate chunk hits across multiple searches.
        merged = self._merge_batches(batches)

        # Rescore chunks using planner-specific logic.
        rescored = self._apply_plan_scores(merged, plan)

        # Enforce diversity so one section does not dominate the final context.
        selected = self._select_diverse_chunks(rescored, plan)

        # Final ordering after planner scoring.
        selected.sort(
            key=lambda item: (
                float(item.metadata.get("planner_score", item.final_score)),
                item.rerank_score,
                item.fused_score,
            ),
            reverse=True,
        )
        return selected

    def _build_factoid_plan(self, *, query: str, scope: QueryScope) -> RetrievalPlan:
        """Plan for narrow factual questions."""
        return RetrievalPlan(
            query=query,
            intent="factoid",
            scope=scope,
            retrieval_queries=[query],
            rerank_instruction="Given a question, judge whether the document chunk directly answers it.",
            candidate_top_k=6,
            final_top_k=5,
            max_chunks_per_heading=3,
            max_chunks_per_page=3,
        )

    def _build_explanation_plan(self, *, query: str, scope: QueryScope) -> RetrievalPlan:
        """Plan for broader explanation-style questions."""
        retrieval_queries = [
            query,
            f"{query} explanation details",
        ]
        return RetrievalPlan(
            query=query,
            intent="explanation",
            scope=scope,
            retrieval_queries=retrieval_queries,
            rerank_instruction=(
                "Given a user request, judge whether the document chunk helps explain the topic clearly and faithfully."
            ),
            candidate_top_k=8,
            final_top_k=6,
            max_chunks_per_heading=2,
            max_chunks_per_page=2,
        )

    def _build_comparison_plan(self, *, query: str, scope: QueryScope) -> RetrievalPlan:
        """Plan for comparison requests."""
        retrieval_queries = [
            query,
            f"{query} differences similarities",
            f"{query} tradeoffs results comparison",
        ]
        return RetrievalPlan(
            query=query,
            intent="comparison",
            scope=scope,
            retrieval_queries=retrieval_queries,
            rerank_instruction=(
                "Given a comparison request, judge whether the document chunk is useful for comparing the requested items."
            ),
            candidate_top_k=8,
            final_top_k=8,
            max_chunks_per_heading=2,
            max_chunks_per_page=2,
        )

    def _build_summary_plan(self, *, query: str, scope: QueryScope) -> RetrievalPlan:
        """
        Plan for document-level summary requests.

        Instead of trusting the raw query alone, we probe multiple summary-relevant
        parts of the document.
        """
        retrieval_queries = [
            query,
            "abstract introduction overview problem statement main contribution",
            "method approach experiment results findings",
            "conclusion discussion limitations future work",
        ]
        return RetrievalPlan(
            query=query,
            intent="broad_summary",
            scope=scope,
            retrieval_queries=retrieval_queries,
            rerank_instruction=(
                "Given a summary request, judge whether the document chunk is useful for a faithful high-level summary."
            ),
            candidate_top_k=10,
            final_top_k=8,
            max_chunks_per_heading=2,
            max_chunks_per_page=2,
            preferred_heading_terms=(
                "abstract",
                "introduction",
                "overview",
                "background",
                "method",
                "approach",
                "results",
                "findings",
                "discussion",
                "conclusion",
                "limitations",
            ),
            discouraged_heading_terms=(
                "references",
                "bibliography",
                "appendix",
                "supplementary",
                "acknowledg",
            ),
        )

    def _search(
        self,
        *,
        index: QdrantHybridIndex,
        query: str,
        doc_ids: list[str] | None,
        top_k: int,
        rerank_instruction: str,
    ) -> list[RetrievedChunk]:
        """
        Run one retrieval call.

        This expects the index.search method to accept a rerank_instruction override.
        """
        return index.search(
            query,
            top_k=top_k,
            doc_ids=doc_ids,
            rerank_instruction=rerank_instruction,
        )

    def _merge_batches(self, batches: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        """
        Merge chunks retrieved by multiple queries.

        If the same chunk appears multiple times, keep the best version and record
        how many retrieval queries found it.
        """
        merged: dict[str, RetrievedChunk] = {}

        for batch in batches:
            for chunk in batch:
                existing = merged.get(chunk.chunk_id)

                if existing is None:
                    merged[chunk.chunk_id] = self._clone_chunk(chunk, query_hits=1)
                    continue

                existing.metadata["planner_query_hits"] = int(
                    existing.metadata.get("planner_query_hits", 1)
                ) + 1

                # If this occurrence has a better score, replace the stored version.
                if chunk.final_score > existing.final_score:
                    merged[chunk.chunk_id] = self._clone_chunk(
                        chunk,
                        query_hits=int(existing.metadata["planner_query_hits"]),
                    )

        return list(merged.values())

    def _apply_plan_scores(
        self,
        chunks: list[RetrievedChunk],
        plan: RetrievalPlan,
    ) -> list[RetrievedChunk]:
        """
        Adjust scores using planner-specific signals.

        Added signals:
        - heading bonus for useful sections
        - penalty for appendix/references
        - bonus when multiple retrieval probes found the same chunk
        - small penalty for picture-only chunks
        """
        rescored: list[RetrievedChunk] = []

        for chunk in chunks:
            metadata = dict(chunk.metadata)

            planner_score = chunk.final_score
            planner_score += self._heading_bonus(metadata=metadata, plan=plan)
            planner_score += self._query_hit_bonus(metadata=metadata)
            planner_score += self._block_type_bonus(metadata=metadata)

            metadata["planner_score"] = planner_score

            rescored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=metadata,
                    source_file=chunk.source_file,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    fused_score=chunk.fused_score,
                    rerank_score=chunk.rerank_score,
                    final_score=planner_score,
                )
            )

        rescored.sort(
            key=lambda item: (item.final_score, item.rerank_score, item.fused_score),
            reverse=True,
        )
        return rescored

    def _select_diverse_chunks(
        self,
        chunks: list[RetrievedChunk],
        plan: RetrievalPlan,
    ) -> list[RetrievedChunk]:
        """
        Keep the final set diverse.

        This prevents top results from being dominated by one heading or one page.
        """
        selected: list[RetrievedChunk] = []
        heading_counts: dict[str, int] = defaultdict(int)
        page_counts: dict[str, int] = defaultdict(int)
        selected_ids: set[str] = set()

        for chunk in chunks:
            heading_bucket = self._heading_bucket(chunk)
            page_bucket = self._page_bucket(chunk)

            if heading_counts[heading_bucket] >= plan.max_chunks_per_heading:
                continue

            if page_counts[page_bucket] >= plan.max_chunks_per_page:
                continue

            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            heading_counts[heading_bucket] += 1
            page_counts[page_bucket] += 1

            if len(selected) >= plan.final_top_k:
                return selected

        # Fallback: if diversity filtering was too strict, fill remaining slots.
        for chunk in chunks:
            if chunk.chunk_id in selected_ids:
                continue
            selected.append(chunk)
            if len(selected) >= plan.final_top_k:
                break

        return selected

    def _heading_bonus(self, *, metadata: dict[str, object], plan: RetrievalPlan) -> float:
        """Boost useful sections and penalize noisy sections."""
        heading_text = self._heading_text(metadata)
        if not heading_text:
            return 0.0

        score = 0.0

        if self._contains_any(heading_text, plan.preferred_heading_terms):
            score += 0.20

        if self._contains_any(heading_text, plan.discouraged_heading_terms):
            score -= 0.35

        return score

    @staticmethod
    def _query_hit_bonus(*, metadata: dict[str, object]) -> float:
        """
        Reward chunks that appear in multiple retrieval probes.

        Multi-hit chunks are often more central to the document.
        """
        hits = int(metadata.get("planner_query_hits", 1))
        if hits <= 1:
            return 0.0
        return min(0.12, 0.04 * (hits - 1))

    @staticmethod
    def _block_type_bonus(*, metadata: dict[str, object]) -> float:
        """Slightly penalize picture-only chunks for text summarization/QA."""
        block_type = str(metadata.get("block_type") or "").strip().lower()
        if block_type == "picture":
            return -0.08
        return 0.0

    def _heading_bucket(self, chunk: RetrievedChunk) -> str:
        """
        Group chunks by heading for diversity control.

        If no heading exists, fall back to page-based grouping.
        """
        headings = chunk.metadata.get("headings")
        if isinstance(headings, list) and headings:
            return self._normalize(str(headings[-1]))
        return self._page_bucket(chunk)

    @staticmethod
    def _page_bucket(chunk: RetrievedChunk) -> str:
        """Group chunks by page when heading metadata is missing."""
        if chunk.page_start is None:
            return f"chunk:{chunk.chunk_id}"
        return f"page:{chunk.page_start}"

    def _heading_text(self, metadata: dict[str, object]) -> str:
        """Flatten heading metadata into one normalized string."""
        headings = metadata.get("headings")
        if not isinstance(headings, list) or not headings:
            return ""
        return self._normalize(" ".join(str(item) for item in headings))

    @staticmethod
    def _clone_chunk(chunk: RetrievedChunk, *, query_hits: int) -> RetrievedChunk:
        """Copy a chunk and attach planner-specific metadata."""
        metadata = dict(chunk.metadata)
        metadata["planner_query_hits"] = query_hits
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=metadata,
            source_file=chunk.source_file,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            fused_score=chunk.fused_score,
            rerank_score=chunk.rerank_score,
            final_score=chunk.final_score,
        )

    @staticmethod
    def _contains_any(text: str, candidates: tuple[str, ...]) -> bool:
        """Return True if any cue phrase appears in the text."""
        return any(candidate in text for candidate in candidates)

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse whitespace for simple rule-based matching."""
        return " ".join(text.lower().split())