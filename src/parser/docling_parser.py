from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from src.config.parser_config import ParserConfig
from src.parser.text_chunk import ParsedChunk


class DoclingParser:
    def __init__(self, config: ParserConfig) -> None:
        config.validate()
        self._config = config
        self.text_converter = config.build_text_converter()
        self._converter = config.build_converter()
        self._tokenizer = config.build_tokenizer()
        self._chunker = HybridChunker(
            tokenizer=self._tokenizer,
            merge_peers=True,
        )

    def parse(
            self,
            source_path: str | Path,
            doc_id: str | None = None,
    ) -> list[ParsedChunk]:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        resolved_doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        document = self._converter.convert(str(path)).document
        text_doc = self.text_converter.convert(str(path)).document

        chunks = self._extract_text_chunks(
            document=text_doc,
            doc_id=resolved_doc_id,
            source_file=path.name,
        )
        chunks = self._merge_small_text_chunks(chunks)

        if self._config.include_picture_chunks:
            picture_chunks = self._extract_picture_chunks(
                document=document,
                doc_id=resolved_doc_id,
                source_file=path.name,
                start_index=len(chunks),
            )
            chunks.extend(picture_chunks)

        return self._reindex_chunks(chunks)

    def _extract_text_chunks(
            self,
            document: Any,
            doc_id: str,
            source_file: str,
    ) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []

        for chunk_index, chunk in enumerate(self._chunker.chunk(dl_doc=document)):
            text = self._chunker.contextualize(chunk=chunk).strip()
            #if chunk_index < 10:
            #    print("=====chunk_index=====", chunk_index)
            #    meta = getattr(chunk, "meta", None)
            #    doc_items = list(getattr(meta, "doc_items", None) or [])
            #    print("====meta======", meta)
            #    print("====doc_items======", doc_items)
            #    print("=====text=====", text)

            if not text:
                continue

            token_count = self._tokenizer.count_tokens(text)
            if token_count > 256:
                print(chunk)
            page_start, page_end = self._extract_chunk_page_range(chunk)
            metadata = self._build_text_chunk_metadata(chunk)
            metadata["token_count"] = token_count

            chunks.append(
                ParsedChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    metadata=metadata,
                )
            )

        return chunks

    def _merge_small_text_chunks(self, chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        if not chunks:
            return []

        merged: list[ParsedChunk] = []
        buffer: ParsedChunk | None = None

        for current in chunks:
            if buffer is None:
                buffer = current
                continue

            buffer_tokens = self._tokenizer.count_tokens(buffer.text)
            current_tokens = self._tokenizer.count_tokens(current.text)
            combined_text = f"{buffer.text}\n\n{current.text}".strip()
            combined_tokens = self._tokenizer.count_tokens(combined_text)

            same_block_type = (
                    buffer.metadata.get("block_type") == "text"
                    and current.metadata.get("block_type") == "text"
            )
            same_headings = (
                    buffer.metadata.get("headings") == current.metadata.get("headings")
            )
            should_merge = (
                    buffer_tokens < self._config.min_chunk_tokens
                    or current_tokens < self._config.min_chunk_tokens
            )

            if (
                    same_block_type
                    and same_headings
                    and should_merge
                    and combined_tokens <= self._config.max_chunk_tokens
            ):
                buffer = ParsedChunk(
                    chunk_id=buffer.chunk_id,
                    doc_id=buffer.doc_id,
                    source_file=buffer.source_file,
                    chunk_index=buffer.chunk_index,
                    page_start=buffer.page_start,
                    page_end=current.page_end,
                    text=combined_text,
                    metadata={
                        **buffer.metadata,
                        "token_count": combined_tokens,
                        "merged_small_chunk": True,
                    },
                )
                continue

            buffer.metadata["token_count"] = buffer_tokens
            merged.append(buffer)
            buffer = current

        if buffer is not None:
            buffer.metadata["token_count"] = self._tokenizer.count_tokens(buffer.text)
            merged.append(buffer)

        return merged

    def _extract_picture_chunks(
            self,
            document: Any,
            doc_id: str,
            source_file: str,
            start_index: int,
    ) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        body_root = getattr(document, "body", None)

        for item, _level in document.iterate_items(
                root=body_root,
                with_groups=False,
                traverse_pictures=True,
        ):
            if not self._is_picture_item(item):
                continue

            text = self._build_picture_text(document=document, picture=item)
            if not text:
                continue

            page_start, page_end = self._extract_item_page_range(item)
            chunk_index = start_index + len(chunks)

            chunks.append(
                ParsedChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    metadata={
                        "parser": "docling",
                        "block_type": "picture",
                        "caption": self._safe_caption_text(document, item),
                        "annotations": self._extract_annotation_texts(item),
                        "token_count": self._tokenizer.count_tokens(text),
                    },
                )
            )

        return chunks

    def _build_picture_text(self, document: Any, picture: Any) -> str:
        annotation_texts = self._extract_annotation_texts(picture)
        caption = self._safe_caption_text(document, picture)

        parts: list[str] = []

        if annotation_texts:
            parts.append("\n".join(annotation_texts).strip())

        if caption:
            normalized_caption = caption.strip()
            annotation_blob = " ".join(annotation_texts).strip().lower()
            if normalized_caption.lower() not in annotation_blob:
                parts.append(normalized_caption)

        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _build_text_chunk_metadata(chunk: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "parser": "docling",
            "block_type": "text",
        }

        meta = getattr(chunk, "meta", None)

        headings = getattr(meta, "headings", None)
        if headings:
            metadata["headings"] = list(headings)

        captions = getattr(meta, "captions", None)
        if captions:
            metadata["captions"] = list(captions)

        return metadata

    @staticmethod
    def _safe_caption_text(document: Any, picture: Any) -> str:
        caption_text = getattr(picture, "caption_text", None)
        if not callable(caption_text):
            return ""

        value = caption_text(document)
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _extract_annotation_texts(picture: Any) -> list[str]:
        get_annotations = getattr(picture, "get_annotations", None)
        if not callable(get_annotations):
            return []

        texts: list[str] = []
        seen: set[str] = set()

        for annotation in get_annotations() or []:
            candidates = (
                getattr(annotation, "text", None),
                getattr(annotation, "content", None),
                getattr(annotation, "description", None),
                getattr(annotation, "body", None),
            )
            for value in candidates:
                if not isinstance(value, str):
                    continue

                normalized = " ".join(value.split()).strip()
                if not normalized:
                    continue

                key = normalized.lower()
                if key in seen:
                    continue

                seen.add(key)
                texts.append(normalized)
                break

        return texts

    @staticmethod
    def _is_picture_item(item: Any) -> bool:
        label = str(getattr(item, "label", "") or "").strip().lower()
        return label == "picture" or item.__class__.__name__ == "PictureItem"

    @staticmethod
    def _extract_chunk_page_range(chunk: Any) -> tuple[int | None, int | None]:
        page_numbers: list[int] = []

        meta = getattr(chunk, "meta", None)
        doc_items = getattr(meta, "doc_items", None) or []

        for item in doc_items:
            prov = getattr(item, "prov", None) or []
            for prov_item in prov:
                page_no = getattr(prov_item, "page_no", None)
                if isinstance(page_no, int):
                    page_numbers.append(page_no)

        if not page_numbers:
            return None, None

        return min(page_numbers), max(page_numbers)

    @staticmethod
    def _extract_item_page_range(item: Any) -> tuple[int | None, int | None]:
        page_numbers: list[int] = []
        prov = getattr(item, "prov", None) or []

        for prov_item in prov:
            page_no = getattr(prov_item, "page_no", None)
            if isinstance(page_no, int):
                page_numbers.append(page_no)

        if not page_numbers:
            return None, None

        return min(page_numbers), max(page_numbers)

    @staticmethod
    def _reindex_chunks(chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        reindexed: list[ParsedChunk] = []

        for index, chunk in enumerate(chunks):
            reindexed.append(
                ParsedChunk(
                    chunk_id=f"{chunk.doc_id}_chunk_{index:04d}",
                    doc_id=chunk.doc_id,
                    source_file=chunk.source_file,
                    chunk_index=index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )

        return reindexed
