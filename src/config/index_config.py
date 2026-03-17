from __future__ import annotations

"""Index configuration for local retrieval models, storage, and ranking limits."""

from dataclasses import dataclass, field
from pathlib import Path

from src.config.model_catalog import ModelCatalog


def _default_project_root() -> Path:
    """Return the repository root based on the config file location."""
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class IndexConfig:
    """Configuration for the hybrid Qdrant retrieval pipeline."""

    project_root: Path = field(default_factory=_default_project_root)
    model_catalog: ModelCatalog = field(default_factory=ModelCatalog)

    qdrant_relative_path: str = "storage/qdrant"
    qdrant_path_override: str | Path | None = None
    collection_name: str = "document_chunks"

    sparse_model_name: str = "Qdrant/bm25"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"

    dense_model_name_override: str | Path | None = None
    reranker_model_name_override: str | Path | None = None

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

    def __post_init__(self) -> None:
        """Normalize paths once so runtime resolution stays deterministic."""
        self.project_root = Path(self.project_root).resolve()

    @property
    def qdrant_path(self) -> str:
        """Return the resolved local Qdrant storage path."""
        if self.qdrant_path_override is not None:
            return str(Path(self.qdrant_path_override).resolve())
        return str((self.project_root / self.qdrant_relative_path).resolve())

    @property
    def dense_model_name(self) -> str:
        """Return the resolved local dense embedder path."""
        if self.dense_model_name_override is not None:
            return str(Path(self.dense_model_name_override).resolve())
        return str(self.model_catalog.embedder_path(self.project_root))

    @property
    def reranker_model_name(self) -> str:
        """Return the resolved local reranker path."""
        if self.reranker_model_name_override is not None:
            return str(Path(self.reranker_model_name_override).resolve())
        return str(self.model_catalog.reranker_path(self.project_root))

    def validate(self) -> None:
        """Validate configuration values and fail early on broken local paths."""
        if not self.collection_name:
            raise ValueError("collection_name must not be empty")
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

        dense_path = Path(self.dense_model_name)
        reranker_path = Path(self.reranker_model_name)

        if not dense_path.exists():
            raise FileNotFoundError(f"Dense model path does not exist: {dense_path}")
        if not reranker_path.exists():
            raise FileNotFoundError(f"Reranker model path does not exist: {reranker_path}")
