from __future__ import annotations

import argparse

from docling.datamodel.base_models import InputFormat

from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser
from src.retrieval.index_config import IndexConfig
from src.retrieval.vector_index import ChunkVectorIndex


def print_results(results: list, preview_chars: int = 700) -> None:
    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        print("=" * 100)
        print(f"rank       : {rank}")
        print(f"score      : {item.score}")
        print(f"chunk_id   : {item.chunk_id}")
        print(f"source_file: {item.source_file}")
        print(f"pages      : {item.page_start} -> {item.page_end}")
        print(f"block_type : {item.metadata.get('block_type')}")
        print(f"headings   : {item.metadata.get('headings')}")
        print("preview:")
        print(item.text[:preview_chars])
        print()


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--pdf", required=True, help="Path to the PDF file")
    cli.add_argument("--query", required=True, help="Query text")
    cli.add_argument("--rebuild", action="store_true")
    args = cli.parse_args()

    parser_config = ParserConfig(
        allowed_formats=[InputFormat.PDF],
        enable_picture_description=True,
        include_picture_chunks=True,
    )

    parser = DoclingParser(parser_config)

    index_config = IndexConfig()
    vector_index = ChunkVectorIndex(index_config)

    chunks = None
    if args.rebuild:
        chunks = parser.parse(args.pdf)

    try:
        index = vector_index.build_or_load(chunks=chunks, rebuild=args.rebuild)
    except FileNotFoundError:
        chunks = parser.parse(args.pdf)
        index = vector_index.build(chunks=chunks)

    results = vector_index.search(
        query=args.query,
        index=index,
    )

    print_results(results)


if __name__ == "__main__":
    main()