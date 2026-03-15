from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class IndexConfig:
    persist_dir: str = "storage/vector_index"
    index_id: str = "document_chunks"
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    similarity_top_k: int = 5
    embed_batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress: bool = True

    def validate(self) -> None:
        if not self.persist_dir:
            raise ValueError("persist_dir must not be empty")
        if not self.index_id:
            raise ValueError("index_id must not be empty")
        if not self.embedding_model_name:
            raise ValueError("embedding_model_name must not be empty")
        if self.similarity_top_k <= 0:
            raise ValueError("similarity_top_k must be greater than 0")
        if self.embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be greater than 0")