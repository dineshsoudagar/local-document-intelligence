"""CLI entrypoint for index lifecycle operations."""

from __future__ import annotations

import argparse

from src.indexing.index_service import IndexService
from src.utils.io import resolve_pdf_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for index management."""
    cli = argparse.ArgumentParser(
        description="Manage the local document index.",
    )
    subparsers = cli.add_subparsers(dest="command", required=True)

    clear_parser = subparsers.add_parser("clear", help="Delete the entire index")
    clear_parser.set_defaults(command="clear")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest one PDF or a folder")
    reindex_parser = subparsers.add_parser("reindex", help="Reindex one PDF or a folder")
    ingest_parser.add_argument("file", help="Path to one PDF file")
    reindex_parser.add_argument("file", help="Path to one PDF file")

    return cli.parse_args()


def main() -> None:
    """Run index management operations."""
    args = parse_args()
    service = IndexService()

    if args.command == "clear":
        service.clear()
        print("Index cleared.")
        return

    if args.command == "ingest":
        path = resolve_pdf_path(args.file)
        doc_id = service.ingest_pdf(path)
        print(f"Ingested: {doc_id}")
        return

    if args.command == "reindex":
        path = resolve_pdf_path(args.file)
        doc_id = service.reindex_pdf(path)
        print(f"Reindexed: {doc_id}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
