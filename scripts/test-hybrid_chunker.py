from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer


def extract_item_summary(item: Any) -> dict[str, Any]:
    prov = getattr(item, "prov", None) or []
    page_numbers: list[int] = []

    for prov_item in prov:
        page_no = getattr(prov_item, "page_no", None)
        if isinstance(page_no, int):
            page_numbers.append(page_no)

    text = (getattr(item, "text", "") or "").strip()
    label = str(getattr(item, "label", "") or "")

    return {
        "class_name": item.__class__.__name__,
        "label": label,
        "text_preview": text[:300],
        "text_length": len(text),
        "pages": sorted(set(page_numbers)),
    }


def build_converter(enable_picture_description: bool) -> DocumentConverter:
    if not enable_picture_description:
        return DocumentConverter()

    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        picture_description_options=PictureDescriptionVlmOptions(
            repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
            prompt="Describe this picture in three to five sentences. Be precise and concise.",
        ),
        generate_picture_images=True,
        images_scale=2.0,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def count_raw_tokens(hf_tokenizer: Any, text: str) -> int:
    try:
        return len(
            hf_tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
        )
    except Exception:
        return -1


def inspect_chunks(
    source_path: str | Path,
    output_path: str | Path = "chunk_debug.json",
    enable_picture_description: bool = True,
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 220,
) -> None:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    converter = build_converter(enable_picture_description=enable_picture_description)

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    document = converter.convert(str(source)).document
    chunks = list(chunker.chunk(dl_doc=document))

    chunk_summaries: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        meta = getattr(chunk, "meta", None)
        doc_items = list(getattr(meta, "doc_items", None) or [])

        try:
            contextualized_text = chunker.contextualize(chunk=chunk).strip()
        except Exception as exc:
            contextualized_text = f"<contextualize failed: {exc}>"

        headings = getattr(meta, "headings", None)
        captions = getattr(meta, "captions", None)

        chunk_summaries.append(
            {
                "chunk_index": index,
                "chunk_class": chunk.__class__.__name__,
                "contextualized_text_preview": contextualized_text[:1000],
                "contextualized_text_length": len(contextualized_text),
                "contextualized_token_count": count_raw_tokens(
                    hf_tokenizer, contextualized_text
                ),
                "meta_class": meta.__class__.__name__ if meta is not None else None,
                "headings": list(headings) if headings else [],
                "captions": list(captions) if captions else [],
                "doc_items_count": len(doc_items),
                "doc_items": [extract_item_summary(item) for item in doc_items],
            }
        )

    payload = {
        "source_file": source.name,
        "enable_picture_description": enable_picture_description,
        "tokenizer_model": tokenizer_model,
        "max_tokens": max_tokens,
        "chunk_count": len(chunks),
        "chunks": chunk_summaries,
    }

    output = Path(output_path)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Wrote chunk inspection to: {output.resolve()}")


if __name__ == "__main__":
    inspect_chunks(
        source_path="../data/pdfs/RAG_survey_paper.pdf",
        output_path="chunk_debug.json",
        enable_picture_description=True,
        tokenizer_model="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=220,
    )