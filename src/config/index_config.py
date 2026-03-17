from __future__ import annotations

"""Index configuration for storage paths, model paths, and retrieval limits."""

from dataclasses import dataclass, field
from pathlib import Path

from src.config.model_catalog import ModelCatalog


def _default_project_root() -> Path:
    """Resolve the repository root from the current config file location."""
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class IndexConfig:
    """Central configuration for the hybrid retrieval index."""
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
        """Normalize the configured project root once at initialization."""
        self.project_root = Path(self.project_root).resolve()

    @property
    def qdrant_path(self) -> str:
        """Resolve the effective Qdrant path, with override support."""
        if self.qdrant_path_override is not None:
            return str(Path(self.qdrant_path_override).resolve())
        return str((self.project_root / self.qdrant_relative_path).resolve())

    @property
    def dense_model_name(self) -> str:
        """Resolve the active dense model path."""
        if self.dense_model_name_override is not None:
            return str(self.dense_model_name_override)
        return self._resolve_model_path("embedder")

    @property
    def reranker_model_name(self) -> str:
        """Resolve the active reranker model path."""
        if self.reranker_model_name_override is not None:
            return str(self.reranker_model_name_override)
        return self._resolve_model_path("reranker")

    def _resolve_model_path(self, key: str) -> str:
        """Resolve a catalog entry into a concrete local model directory."""
        entry = self.model_catalog.get(key)
        return str(
            entry.resolve_dir(
                project_root=self.project_root,
                models_root=self.model_catalog.models_root,
            )
        )

    def validate(self) -> None:
        """Fail fast on invalid index configuration."""
        if not self.collection_name:
            raise ValueError("collection_name must not be empty")
        if not self.sparse_model_name:
            raise ValueError("sparse_model_name must not be empty")
        if not self.qdrant_path:
            raise ValueError("qdrant_path must not be empty")
        if not self.dense_model_name:
            raise ValueError("dense_model_name must not be empty")
        if not self.reranker_model_name:
            raise ValueError("reranker_model_name must not be empty")
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
