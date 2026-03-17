from __future__ import annotations

"""CLI entrypoint for indexing a PDF and running hybrid search."""

import argparse
from pathlib import Path
import time

from docling.datamodel.base_models import InputFormat

from src.config.index_config import IndexConfig
from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex
import hashlib

def print_results(results: list, preview_chars: int | None = None) -> None:
    """Print final retrieval results in a compact debug-friendly format."""
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
    """Build a stable document identifier from filename and file bytes."""
    pdf_path = Path(path).resolve()
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:12]
    stem = pdf_path.stem.strip().replace(" ", "_").replace("-", "_").lower()
    return f"{stem}_{digest}"


def main() -> None:
    """Build the index if needed, then run normal or debug search."""
    cli = argparse.ArgumentParser()
    cli.add_argument("--pdf", required=True, help="Path to the PDF file")
    cli.add_argument("--query", required=True, help="Query text")
    cli.add_argument("--rebuild", action="store_true")
    cli.add_argument("--debug", action="store_true")
    args = cli.parse_args()

    # Parser setup for PDF ingestion.
    parser_config = ParserConfig(
        allowed_formats=[InputFormat.PDF],
        enable_picture_description=False,
        include_picture_chunks=True,
    )
    parser = DoclingParser(parser_config)

    # Retrieval engine owns Qdrant, dense embeddings, and reranking.
    index = QdrantHybridIndex(IndexConfig())

    # Only parse and build when requested or when no collection exists yet.
    if args.rebuild or not index.collection_exists():
        chunks = parser.parse(args.pdf, doc_id=build_doc_id(args.pdf))
        index.build(chunks=chunks, rebuild=args.rebuild)

    search_start = time.perf_counter()

    # Debug mode exposes dense, sparse, fused, and reranked stages separately.
    if args.debug:
        debug = index.debug_search(args.query)

        for stage_name in ["dense", "sparse", "fused", "reranked"]:
            print(f"\n{'#' * 30} {stage_name.upper()} {'#' * 30}")
            for rank, item in enumerate(debug[stage_name][:10], start=1):
                print("-" * 100)
                print(f"rank    : {rank}")
                print(f"chunk_id : {item['chunk_id']}")
                print(f"score    : {item.get('score', item.get('fusion_score'))}")
                if "rerank_score" in item:
                    print(f"rerank   : {item['rerank_score']}")
                    print(f"fusion   : {item['fusion_score']}")
                print(f"pages    : {item['pages']}")
                print(f"headings : {item['headings']}")
                print("preview:")
                print(item["preview"])

        search_elapsed = time.perf_counter() - search_start
        print(f"\nsearch_time_seconds: {search_elapsed:.3f}")
        return

    results = index.search(query=args.query)
    print_results(results)

    search_elapsed = time.perf_counter() - search_start
    print(f"\nsearch_time_seconds: {search_elapsed:.3f}")


if __name__ == "__main__":
    main()
