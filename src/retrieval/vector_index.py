from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.parser.text_chunk import ParsedChunk
from src.config.index_config import IndexConfig


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    score: float | None
    text: str
    metadata: dict
    source_file: str | None
    page_start: int | None
    page_end: int | None


class ChunkVectorIndex:
    def __init__(self, config: IndexConfig) -> None:
        config.validate()
        self._config = config
        self._embed_model = HuggingFaceEmbedding(
            model_name=config.embedding_model_name,
            embed_batch_size=config.embed_batch_size,
            normalize=config.normalize_embeddings,
        )
        Settings.embed_model = self._embed_model

    def build(self, chunks: list[ParsedChunk], rebuild: bool = False) -> VectorStoreIndex:
        if rebuild:
            self._reset_storage()

        nodes = [self._chunk_to_node(chunk) for chunk in chunks if chunk.text.strip()]
        if not nodes:
            raise ValueError("No non-empty chunks were provided for indexing")

        index = VectorStoreIndex(nodes, show_progress=self._config.show_progress)
        index.set_index_id(self._config.index_id)
        index.storage_context.persist(persist_dir=self._config.persist_dir)
        return index

    def load(self) -> VectorStoreIndex:
        persist_path = Path(self._config.persist_dir)
        if not persist_path.exists():
            raise FileNotFoundError(
                f"Persisted index directory does not exist: {persist_path.resolve()}"
            )

        storage_context = StorageContext.from_defaults(
            persist_dir=self._config.persist_dir
        )
        return load_index_from_storage(
            storage_context=storage_context,
            index_id=self._config.index_id,
        )

    def build_or_load(
        self,
        chunks: list[ParsedChunk] | None = None,
        rebuild: bool = False,
    ) -> VectorStoreIndex:
        persist_path = Path(self._config.persist_dir)

        if rebuild or not persist_path.exists():
            if chunks is None:
                raise ValueError("chunks are required when building a new index")
            return self.build(chunks=chunks, rebuild=rebuild)

        return self.load()

    def search(
        self,
        query: str,
        index: VectorStoreIndex,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        similarity_top_k = top_k or self._config.similarity_top_k
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        results = retriever.retrieve(query)

        retrieved: list[RetrievedChunk] = []
        for result in results:
            node = result.node
            metadata = dict(node.metadata or {})
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id") or getattr(node, "node_id", "")),
                    score=float(result.score) if result.score is not None else None,
                    text=node.get_content(metadata_mode=MetadataMode.NONE),
                    metadata=metadata,
                    source_file=metadata.get("source_file"),
                    page_start=metadata.get("page_start"),
                    page_end=metadata.get("page_end"),
                )
            )

        return retrieved

    def _chunk_to_node(self, chunk: ParsedChunk) -> TextNode:
        metadata = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            **chunk.metadata,
        }
        excluded_metadata_keys = list(metadata.keys())

        return TextNode(
            id_=chunk.chunk_id,
            text=chunk.text,
            metadata=metadata,
            excluded_embed_metadata_keys=excluded_metadata_keys,
            excluded_llm_metadata_keys=excluded_metadata_keys,
        )

    def _reset_storage(self) -> None:
        persist_path = Path(self._config.persist_dir)
        if persist_path.exists():
            shutil.rmtree(persist_path)