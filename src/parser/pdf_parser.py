from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# LlamaIndex reader package:
# pip install llama-index llama-index-readers-file pypdf
from llama_index.readers.file import PyMuPDFReader


@dataclass
class ParsedChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page_label: str | None
    chunk_index: int
    text: str
    metadata: Dict[str, Any]


class PDFChunkParser:
    """
    Simple PDF -> cleaned text -> LlamaIndex nodes -> serializable chunks.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.reader = PyMuPDFReader()

    def load_pdf_pages(self, pdf_path: str | Path) -> List[Document]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        docs = self.reader.load_data(file_path=str(pdf_path))
        # PyMuPDFReader usually returns one Document per page with metadata.
        return docs

    def clean_text(self, text: str) -> str:
        """
        Minimal cleaning only.
        Do not get cute here yet.
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace around line breaks
        text = re.sub(r"[ \t]+\n", "\n", text)

        # Collapse 3+ blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Merge single newlines inside paragraphs, preserve paragraph breaks
        # Example:
        # "RAG systems are useful.\nThey help..." -> same paragraph
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Collapse repeated spaces
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text.strip()

    def build_document_from_pages(
        self,
        page_docs: List[Document],
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> Document:
        pdf_path = Path(pdf_path)
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"

        cleaned_pages = []
        page_map = []

        for i, page_doc in enumerate(page_docs):
            raw_text = page_doc.text or ""
            cleaned = self.clean_text(raw_text)
            if not cleaned:
                continue

            page_num = page_doc.metadata.get("page", i + 1)
            cleaned_pages.append(cleaned)

            # Keep a lightweight page trace
            page_map.append(
                {
                    "page_num": page_num,
                    "char_len": len(cleaned),
                }
            )

        full_text = "\n\n".join(cleaned_pages)

        metadata = {
            "doc_id": doc_id,
            "source_file": pdf_path.name,
            "source_path": str(pdf_path.resolve()),
            "num_pages_loaded": len(page_docs),
            "num_pages_after_cleaning": len(cleaned_pages),
            "page_map": page_map,
        }

        return Document(text=full_text, metadata=metadata)

    def chunk_pdf(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> List[ParsedChunk]:
        page_docs = self.load_pdf_pages(pdf_path)
        merged_doc = self.build_document_from_pages(page_docs, pdf_path, doc_id=doc_id)

        nodes = self.splitter.get_nodes_from_documents([merged_doc])

        chunks: List[ParsedChunk] = []
        for idx, node in enumerate(nodes):
            node_meta = dict(node.metadata) if node.metadata else {}

            chunk = ParsedChunk(
                chunk_id=f"{node_meta.get('doc_id', 'doc')}_chunk_{idx:04d}",
                doc_id=node_meta.get("doc_id", "unknown_doc"),
                source_file=node_meta.get("source_file", Path(pdf_path).name),
                page_label=None,  # We can improve this later with page-range mapping
                chunk_index=idx,
                text=node.text,
                metadata=node_meta,
            )
            chunks.append(chunk)

        return chunks


def inspect_chunks(chunks: List[ParsedChunk], limit: int = 5, preview_chars: int = 700) -> None:
    for chunk in chunks[:limit]:
        print("=" * 100)
        print(f"chunk_id   : {chunk.chunk_id}")
        print(f"source_file: {chunk.source_file}")
        print(f"chunk_index: {chunk.chunk_index}")
        print(f"text_len   : {len(chunk.text)}")
        print("preview:")
        print(chunk.text[:preview_chars])
        print()


def chunks_to_dicts(chunks: List[ParsedChunk]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in chunks]


if __name__ == "__main__":
    parser = PDFChunkParser(chunk_size=512, chunk_overlap=64)

    pdf_path = "sample.pdf"   # replace with your file
    chunks = parser.chunk_pdf(pdf_path)

    print(f"Total chunks: {len(chunks)}")
    inspect_chunks(chunks, limit=5)

    # Example: serialize for later indexing/storage
    chunk_dicts = chunks_to_dicts(chunks)
    print(f"First metadata keys: {list(chunk_dicts[0]['metadata'].keys()) if chunk_dicts else []}")