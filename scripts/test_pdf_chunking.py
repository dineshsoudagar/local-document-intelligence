from __future__ import annotations

import argparse
from pathlib import Path

from src.config.parser_config import ParserConfig
from src.parser.pdf_parser import PDFParser
from src.utils.io import write_json


def inspect_chunks(chunks: list, limit: int = 5, preview_chars: int = 700) -> None:
    for chunk in chunks[:limit]:
        print("=" * 100)
        print(f"chunk_id   : {chunk.chunk_id}")
        print(f"source_file: {chunk.source_file}")
        print(f"chunk_index: {chunk.chunk_index}")
        print(f"text_len   : {len(chunk.text)}")
        print("preview:")
        print(chunk.text[:preview_chars])
        print()


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--pdf", required=True, help="Path to the PDF file")
    cli.add_argument("--chunk-size", type=int, default=512)
    cli.add_argument("--chunk-overlap", type=int, default=64)
    cli.add_argument("--inspect-limit", type=int, default=5)
    cli.add_argument("--output-json", default=None)

    args = cli.parse_args()

    config = ParserConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    parser = PDFParser(config=config)
    chunks = parser.parse(args.pdf)

    if not args.output_json:
        print(f"total_chunks: {len(chunks)}")
        inspect_chunks(chunks, limit=args.inspect_limit)

    if args.output_json:
        write_json([chunk.to_dict() for chunk in chunks], args.output_json)
        print(f"saved_json: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
