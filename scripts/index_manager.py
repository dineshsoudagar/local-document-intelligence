from __future__ import annotations

"""CLI entrypoint for index lifecycle operations."""

import argparse

from src.indexing.index_service import IndexService


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for index management."""
    cli = argparse.ArgumentParser(
        description="Manage the local document index.",
    )
    subparsers = cli.add_subparsers(dest="command", required=True)

    clear_parser = subparsers.add_parser("clear", help="Delete the entire index")
    clear_parser.set_defaults(command="clear")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest one PDF or a folder")
    ingest_parser.add_argument("--pdf", default=None, help="Path to one PDF file")
    ingest_parser.add_argument("--folder", default=None, help="Path to a folder of PDFs")

    reindex_parser = subparsers.add_parser("reindex", help="Reindex one PDF or a folder")
    reindex_parser.add_argument("--pdf", default=None, help="Path to one PDF file")
    reindex_parser.add_argument("--folder", default=None, help="Path to a folder of PDFs")

    return cli.parse_args()


def _validate_source_args(pdf: str | None, folder: str | None) -> None:
    if bool(pdf) == bool(folder):
        raise ValueError("Provide exactly one of --pdf or --folder")


def main() -> None:
    """Run index management operations."""
    args = parse_args()
    service = IndexService()

    if args.command == "clear":
        service.clear()
        print("Index cleared.")
        return

    if args.command == "ingest":
        _validate_source_args(args.pdf, args.folder)
        if args.pdf:
            doc_id = service.ingest_pdf(args.pdf)
            print(f"Ingested: {doc_id}")
            return
        doc_ids = service.ingest_folder(args.folder)
        print(f"Ingested {len(doc_ids)} document(s).")
        return

    if args.command == "reindex":
        _validate_source_args(args.pdf, args.folder)
        if args.pdf:
            doc_id = service.reindex_pdf(args.pdf)
            print(f"Reindexed: {doc_id}")
            return
        doc_ids = service.reindex_folder(args.folder)
        print(f"Reindexed {len(doc_ids)} document(s).")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()