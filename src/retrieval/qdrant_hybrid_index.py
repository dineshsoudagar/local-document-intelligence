"""Hybrid Qdrant index with dense retrieval, sparse retrieval, and reranking."""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from dataclasses import dataclass, replace
from typing import Any, Sequence
from src.config.index_config import IndexConfig
from src.parser.text_chunk import ParsedChunk
from src.retrieval.qwen_models import QwenDenseEmbedder, QwenReranker


@dataclass(slots=True)
class RetrievedChunk:
    """Normalized final ranked result returned by the retriever."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    source_file: str | None
    page_start: int | None
    page_end: int | None
    fused_score: float
    rerank_score: float
    final_score: float


class QdrantHybridIndex:
    """Own the local Qdrant collection and the hybrid retrieval pipeline."""

    def __init__(self, config: IndexConfig) -> None:
        config.validate()
        self._config = config

        Path(self._config.qdrant_path).mkdir(parents=True, exist_ok=True)

        self._client = QdrantClient(path=self._config.qdrant_path)
        self._embedder = QwenDenseEmbedder(
            model_name=self._config.dense_model_name,
            batch_size=self._config.dense_batch_size,
            show_progress=self._config.show_progress,
        )
        self._reranker: QwenReranker | None = None

    def collection_exists(self) -> bool:
        """Return whether the configured collection already exists."""
        return self._client.collection_exists(self._config.collection_name)

    def build(self, chunks: list[ParsedChunk], rebuild: bool = False) -> None:
        """Create or refresh the collection and upsert parsed chunks."""
        filtered_chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not filtered_chunks:
            raise ValueError("No non-empty chunks were provided for indexing")

        if rebuild and self.collection_exists():
            self._client.delete_collection(self._config.collection_name)

        if not self.collection_exists():
            self._create_collection()

        texts = [chunk.text for chunk in filtered_chunks]
        avg_doc_len = self._average_document_length(texts)

        for start in range(0, len(filtered_chunks), self._config.upsert_batch_size):
            batch_chunks = filtered_chunks[start: start + self._config.upsert_batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_dense_vectors = self._embedder.encode_documents(batch_texts)

            points = [
                self._build_point(
                    chunk=chunk,
                    dense_vector=dense_vector,
                    avg_doc_len=avg_doc_len,
                )
                for chunk, dense_vector in zip(batch_chunks, batch_dense_vectors)
            ]
            self._client.upsert(
                collection_name=self._config.collection_name,
                points=points,
                wait=True,
            )

    def _build_point(
            self,
            chunk: ParsedChunk,
            dense_vector: list[float],
            avg_doc_len: float,
    ) -> models.PointStruct:
        """Convert one parsed chunk into one Qdrant point."""
        return models.PointStruct(
            id=self._build_point_id(chunk),
            payload=self._build_payload(chunk),
            vector={
                self._config.dense_vector_name: dense_vector,
                self._config.sparse_vector_name: models.SparseVector(
                    indices=[],
                    values=[],
                ),
            },
        )

    @staticmethod
    def _build_point_id(chunk: ParsedChunk) -> str:
        """Return a stable UUID for a parsed chunk."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

    @staticmethod
    def _build_payload(chunk: ParsedChunk) -> dict[str, Any]:
        """Store text and metadata needed for later inspection and citation."""
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "text": chunk.text,
            **chunk.metadata,
        }

    def _build_doc_filter(self, doc_ids: list[str] | None) -> models.Filter | None:
        """Filter the retrieval only to the input doc_ids"""
        if not doc_ids:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=doc_ids),
                )
            ]
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        rerank_instruction: str | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid retrieval, rerank fused candidates, and return final results."""
        fused_limit = self._config.fused_top_k
        final_limit = top_k or self._config.final_top_k

        instruction = rerank_instruction or self._config.rerank_instruction

        dense_query = self._embedder.encode_query(query)
        sparse_query = models.Document(
            text=query,
            model=self._config.sparse_model_name,
        )

        doc_filter = self._build_doc_filter(doc_ids)

        response = self._client.query_points(
            collection_name=self._config.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_query,
                    using=self._config.sparse_vector_name,
                    limit=self._config.sparse_top_k,
                    filter=doc_filter,
                ),
                models.Prefetch(
                    query=dense_query,
                    using=self._config.dense_vector_name,
                    limit=self._config.dense_top_k,
                    filter=doc_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fused_limit,
            with_payload=True,
            query_filter=doc_filter,
        )

        fused_candidates = [self._scored_point_to_candidate(point) for point in response.points]
        if not fused_candidates:
            return []

        reranker = self._get_reranker()

        rerank_scores = reranker.score(
            query=query,
            documents=[candidate["text"] for candidate in fused_candidates],
            instruction=instruction,
        )

        if len(rerank_scores) != len(fused_candidates):
            raise RuntimeError(
                "Reranker returned a mismatched number of scores: "
                f"{len(rerank_scores)} != {len(fused_candidates)}"
            )


        results: list[RetrievedChunk] = []
        for candidate, rerank_score in zip(fused_candidates, rerank_scores):
            results.append(
                RetrievedChunk(
                    chunk_id=candidate["chunk_id"],
                    text=candidate["text"],
                    metadata=candidate["metadata"],
                    source_file=candidate["source_file"],
                    page_start=candidate["page_start"],
                    page_end=candidate["page_end"],
                    fused_score=candidate["fused_score"],
                    rerank_score=rerank_score,
                    final_score=rerank_score,
                )
            )

        results.sort(
            key=lambda item: (item.rerank_score, item.fused_score),
            reverse=True,
        )
        return results[:final_limit]

    def debug_search(self, query: str, max_text_len: int = -1) -> dict[str, Any]:
        """Expose dense, sparse, fused, and reranked stages for inspection."""
        dense_query = self._embedder.encode_query(query)
        sparse_query = models.Document(
            text=query,
            model=self._config.sparse_model_name,
        )

        dense_response = self._client.query_points(
            collection_name=self._config.collection_name,
            query=dense_query,
            using=self._config.dense_vector_name,
            limit=self._config.dense_top_k,
            with_payload=True,
        )

        sparse_response = self._client.query_points(
            collection_name=self._config.collection_name,
            query=sparse_query,
            using=self._config.sparse_vector_name,
            limit=self._config.sparse_top_k,
            with_payload=True,
        )

        fused_response = self._client.query_points(
            collection_name=self._config.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_query,
                    using=self._config.sparse_vector_name,
                    limit=self._config.sparse_top_k,
                ),
                models.Prefetch(
                    query=dense_query,
                    using=self._config.dense_vector_name,
                    limit=self._config.dense_top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self._config.fused_top_k,
            with_payload=True,
        )

        fused_candidates = [self._scored_point_to_candidate(point) for point in fused_response.points]

        reranker = self._get_reranker()
        rerank_scores = reranker.score(
            query=query,
            documents=[candidate["text"] for candidate in fused_candidates],
            instruction=self._config.rerank_instruction,
        )

        if len(rerank_scores) != len(fused_candidates):
            raise RuntimeError(
                "Reranker returned a mismatched number of scores: "
                f"{len(rerank_scores)} != {len(fused_candidates)}"
            )

        reranked: list[dict[str, Any]] = []
        for candidate, rerank_score in zip(fused_candidates, rerank_scores):
            reranked.append(
                {
                    "chunk_id": candidate["chunk_id"],
                    "headings": candidate["metadata"].get("headings"),
                    "pages": (candidate["page_start"], candidate["page_end"]),
                    "fusion_score": candidate["fused_score"],
                    "rerank_score": rerank_score,
                    "final_score": rerank_score,
                    "preview": candidate["text"][:max_text_len],
                }
            )

        reranked.sort(
            key=lambda item: (item["rerank_score"], item["fusion_score"]),
            reverse=True,
        )

        return {
            "dense": [
                {
                    "chunk_id": str((point.payload or {}).get("chunk_id") or point.id),
                    "score": point.score,
                    "headings": (point.payload or {}).get("headings"),
                    "pages": ((point.payload or {}).get("page_start"), (point.payload or {}).get("page_end")),
                    "preview": str((point.payload or {}).get("text", ""))[:max_text_len],
                }
                for point in dense_response.points
            ],
            "sparse": [
                {
                    "chunk_id": str((point.payload or {}).get("chunk_id") or point.id),
                    "score": point.score,
                    "headings": (point.payload or {}).get("headings"),
                    "pages": ((point.payload or {}).get("page_start"), (point.payload or {}).get("page_end")),
                    "preview": str((point.payload or {}).get("text", ""))[:max_text_len],
                }
                for point in sparse_response.points
            ],
            "fused": [
                {
                    "chunk_id": candidate["chunk_id"],
                    "headings": candidate["metadata"].get("headings"),
                    "pages": (candidate["page_start"], candidate["page_end"]),
                    "fusion_score": candidate["fused_score"],
                    "preview": candidate["text"][:max_text_len],
                }
                for candidate in fused_candidates
            ],
            "reranked": reranked,
        }

    def _create_collection(self) -> None:
        """Create the hybrid collection schema used by the retriever."""
        self._client.create_collection(
            collection_name=self._config.collection_name,
            vectors_config={
                self._config.dense_vector_name: models.VectorParams(
                    size=self._embedder.dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self._config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    @staticmethod
    def _average_document_length(texts: list[str]) -> float:
        """Estimate average token-like document length for sparse indexing."""
        if not texts:
            return 0.0
        lengths = [len(text.split()) for text in texts]
        return sum(lengths) / len(lengths)

    def _scored_point_to_candidate(self, point: models.ScoredPoint) -> dict[str, Any]:
        """Convert a raw fused Qdrant point into a validated candidate payload."""
        if point.score is None:
            raise RuntimeError("Qdrant returned a fused candidate without a score")

        payload = dict(point.payload or {})
        text = str(payload.pop("text", ""))
        chunk_id = str(payload.get("chunk_id") or point.id)

        return {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": payload,
            "source_file": payload.get("source_file"),
            "page_start": payload.get("page_start"),
            "page_end": payload.get("page_end"),
            "fused_score": float(point.score),
        }

    def _get_reranker(self) -> QwenReranker:
        """Load the reranker lazily and reuse it across searches."""
        if self._reranker is None:
            self._reranker = QwenReranker(
                model_name=self._config.reranker_model_name,
                batch_size=self._config.rerank_batch_size,
                max_length=self._config.reranker_max_length,
            )
        return self._reranker

    def rerank_existing_chunks(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        *,
        instruction: str | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank an existing chunk set against one common query."""
        if not chunks:
            return []

        reranker = self._get_reranker()
        rerank_instruction = instruction or self._config.rerank_instruction
        rerank_scores = reranker.score(
            query=query,
            documents=[chunk.text for chunk in chunks],
            instruction=rerank_instruction,
        )

        if len(rerank_scores) != len(chunks):
            raise RuntimeError(
                "Reranker returned a mismatched number of scores: "
                f"{len(rerank_scores)} != {len(chunks)}"
            )

        reranked = [
            replace(
                chunk,
                rerank_score=score,
                final_score=score,
            )
            for chunk, score in zip(chunks, rerank_scores)
        ]
        reranked.sort(
            key=lambda item: (item.rerank_score, item.fused_score),
            reverse=True,
        )
        return reranked


    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        """Scale values into the range [0, 1]."""
        if not values:
            return []

        min_value = min(values)
        max_value = max(values)

        if max_value == min_value:
            return [1.0] * len(values)

        scale = max_value - min_value
        return [(value - min_value) / scale for value in values]

    def close(self) -> None:
        self._client.close()

    def clear(self) -> None:
        """Delete the configured collection if it exists."""
        if self.collection_exists():
            self._client.delete_collection(self._config.collection_name)

    def document_exists(self, doc_id: str) -> bool:
        """Return whether any chunk for the document exists in the collection."""
        if not self.collection_exists():
            return False

        response = self._client.scroll(
            collection_name=self._config.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        points, _ = response
        return bool(points)

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to one document."""
        if not self.collection_exists():
            return

        self._client.delete(
            collection_name=self._config.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
