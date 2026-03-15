from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IndexConfig:
    qdrant_path: str = "storage/qdrant"
    collection_name: str = "document_chunks"

    dense_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    sparse_model_name: str = "Qdrant/bm25"

    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"

    dense_query_instruction: str = (
        "Given a document search query, retrieve relevant passages that answer the query"
    )
    rerank_instruction: str = (
        "Given a document search query, judge whether the document answers the query"
    )

    dense_batch_size: int = 8
    upsert_batch_size: int = 64
    rerank_batch_size: int = 4
    reranker_max_length: int = 4096

    dense_top_k: int = 20
    sparse_top_k: int = 20
    fused_top_k: int = 30
    final_top_k: int = 5

    show_progress: bool = True

    def validate(self) -> None:
        if not self.qdrant_path:
            raise ValueError("qdrant_path must not be empty")
        if not self.collection_name:
            raise ValueError("collection_name must not be empty")
        if not self.dense_model_name:
            raise ValueError("dense_model_name must not be empty")
        if not self.reranker_model_name:
            raise ValueError("reranker_model_name must not be empty")
        if not self.sparse_model_name:
            raise ValueError("sparse_model_name must not be empty")
        if self.dense_batch_size <= 0:
            raise ValueError("dense_batch_size must be greater than 0")
        if self.upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be greater than 0")
        if self.rerank_batch_size <= 0:
            raise ValueError("rerank_batch_size must be greater than 0")
        if self.reranker_max_length <= 0:
            raise ValueError("reranker_max_length must be greater than 0")
        if self.dense_top_k <= 0:
            raise ValueError("dense_top_k must be greater than 0")
        if self.sparse_top_k <= 0:
            raise ValueError("sparse_top_k must be greater than 0")
        if self.fused_top_k <= 0:
            raise ValueError("fused_top_k must be greater than 0")
        if self.final_top_k <= 0:
            raise ValueError("final_top_k must be greater than 0")
        if self.final_top_k > self.fused_top_k:
            raise ValueError("final_top_k must be smaller than or equal to fused_top_k")