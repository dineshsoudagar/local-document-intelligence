from __future__ import annotations

"""Hybrid Qdrant index with dense retrieval, sparse retrieval, and reranking."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from src.config.index_config import IndexConfig
from src.parser.text_chunk import ParsedChunk
from src.retrieval.qwen_models import QwenDenseEmbedder, QwenReranker


@dataclass(slots=True)
class RetrievedChunk:
    """Normalized search result returned by the retriever."""
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    source_file: str | None
    page_start: int | None
    page_end: int | None
    fused_score: float | None
    rerank_score: float | None = None


class QdrantHybridIndex:
    """Own the local Qdrant collection and the hybrid retrieval pipeline."""

    def __init__(self, config: IndexConfig) -> None:
        config.validate()
        self._config = config

        # Ensure local storage exists before the client is initialized.
        Path(self._config.qdrant_path).mkdir(parents=True, exist_ok=True)

        self._client = QdrantClient(path=self._config.qdrant_path)
        self._embedder = QwenDenseEmbedder(
            model_name=self._config.dense_model_name,
            batch_size=self._config.dense_batch_size,
            show_progress=self._config.show_progress,
        )

        # Reranker is loaded only when search needs it.
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

        # Embeddings and upserts run in batches to keep memory predictable.
        for start in range(0, len(filtered_chunks), self._config.upsert_batch_size):
            batch_chunks = filtered_chunks[start : start + self._config.upsert_batch_size]
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
                # The sparse side is stored from raw text using the configured sparse model.
                self._config.sparse_vector_name: models.Document(
                    text=chunk.text,
                    model=self._config.sparse_model_name,
                    options={"avg_len": avg_doc_len},
                ),
            },
        )

    @staticmethod
    def _build_point_id(chunk: ParsedChunk) -> str:
        """Build a deterministic point ID from the chunk identifier."""
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

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Run hybrid retrieval, then rerank fused candidates."""
        fused_limit = self._config.fused_top_k
        final_limit = top_k or self._config.final_top_k

        # Dense and sparse query forms are prepared separately.
        dense_query = self._embedder.encode_query(query)
        sparse_query = models.Document(
            text=query,
            model=self._config.sparse_model_name,
        )

        # Prefetch runs the dense and sparse searches first.
        # FusionQuery then combines both ranked lists with RRF.
        response = self._client.query_points(
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
            limit=fused_limit,
            with_payload=True,
        )

        candidates = [self._scored_point_to_result(point) for point in response.points]
        if not candidates:
            return []

        reranker = self._get_reranker()
        rerank_scores = reranker.score(
            query=query,
            documents=[candidate.text for candidate in candidates],
            instruction=self._config.rerank_instruction,
        )

        rescored: list[RetrievedChunk] = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            rescored.append(
                RetrievedChunk(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    source_file=candidate.source_file,
                    page_start=candidate.page_start,
                    page_end=candidate.page_end,
                    fused_score=candidate.fused_score,
                    rerank_score=rerank_score,
                )
            )

        # Final ordering prefers reranker confidence, then fusion score as tie-breaker.
        rescored.sort(
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else float("-inf"),
                item.fused_score if item.fused_score is not None else float("-inf"),
            ),
            reverse=True,
        )
        return rescored[:final_limit]

    def debug_search(self, query: str, max_text_len: int = -1) -> dict[str, list[dict[str, Any]]]:
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

        fused_candidates = [self._scored_point_to_result(point) for point in fused_response.points]

        reranker = self._get_reranker()
        rerank_scores = reranker.score(
            query=query,
            documents=[candidate.text for candidate in fused_candidates],
            instruction=self._config.rerank_instruction,
        )

        reranked = []
        for candidate, rerank_score in zip(fused_candidates, rerank_scores):
            reranked.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "headings": candidate.metadata.get("headings"),
                    "pages": (candidate.page_start, candidate.page_end),
                    "fusion_score": candidate.fused_score,
                    "rerank_score": rerank_score,
                    "preview": candidate.text[:max_text_len],
                }
            )

        reranked.sort(
            key=lambda item: (
                item["rerank_score"] if item["rerank_score"] is not None else float("-inf"),
                item["fusion_score"] if item["fusion_score"] is not None else float("-inf"),
            ),
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
                    "chunk_id": candidate.chunk_id,
                    "headings": candidate.metadata.get("headings"),
                    "pages": (candidate.page_start, candidate.page_end),
                    "fusion_score": candidate.fused_score,
                    "preview": candidate.text[:max_text_len],
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

    def _scored_point_to_result(self, point: models.ScoredPoint) -> RetrievedChunk:
        """Convert a raw Qdrant point into the project result shape."""
        payload = dict(point.payload or {})
        text = str(payload.pop("text", ""))
        chunk_id = str(payload.get("chunk_id") or point.id)

        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=payload,
            source_file=payload.get("source_file"),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
            fused_score=float(point.score) if point.score is not None else None,
        )

    def _get_reranker(self) -> QwenReranker:
        """Load the reranker lazily and reuse it across searches."""
        if self._reranker is None:
            self._reranker = QwenReranker(
                model_name=self._config.reranker_model_name,
                batch_size=self._config.rerank_batch_size,
                max_length=self._config.reranker_max_length,
            )
        return self._reranker
