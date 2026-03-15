from __future__ import annotations

import argparse
from pathlib import Path

from docling.datamodel.base_models import InputFormat

from src.config.index_config import IndexConfig
from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex


def print_results(results: list, preview_chars: int | None = None) -> None:
    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        print("=" * 100)
        print(f"rank         : {rank}")
        print(f"rerank_score : {item.rerank_score}")
        print(f"fusion_score : {item.fused_score}")
        print(f"chunk_id     : {item.chunk_id}")
        print(f"source_file  : {item.source_file}")
        print(f"pages        : {item.page_start} -> {item.page_end}")
        print(f"block_type   : {item.metadata.get('block_type')}")
        print(f"headings     : {item.metadata.get('headings')}")
        print("preview:")
        if preview_chars:
            print(item.text[:preview_chars])
        else:
            print(item.text)
        print()


def build_doc_id(path: str) -> str:
    pdf_path = Path(path)
    return pdf_path.stem.strip().replace(" ", "_").replace("-", "_").lower()


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--pdf", required=True, help="Path to the PDF file")
    cli.add_argument("--query", required=True, help="Query text")
    cli.add_argument("--rebuild", action="store_true")
    args = cli.parse_args()

    parser_config = ParserConfig(
        allowed_formats=[InputFormat.PDF],
        enable_picture_description=False,
        include_picture_chunks=True,
    )
    parser = DoclingParser(parser_config)

    index = QdrantHybridIndex(IndexConfig())

    if args.rebuild or not index.collection_exists():
        chunks = parser.parse(args.pdf, doc_id=build_doc_id(args.pdf))
        index.build(chunks=chunks, rebuild=args.rebuild)

    results = index.search(query=args.query)
    print_results(results)


if __name__ == "__main__":
    main()
