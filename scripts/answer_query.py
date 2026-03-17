from __future__ import annotations

from pathlib import Path

from src.utils.io import resolve_pdf_path

"""CLI entrypoint for grounded answer generation over the local corpus."""

import argparse
import json

from src.config.generator_config import GeneratorConfig
from src.generation.answer_service import GroundedAnswerService
from src.generation.context_builder import render_sources
from src.indexing.index_service import IndexService


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for grounded answering."""
    cli = argparse.ArgumentParser(
        description="Answer a query from the local indexed corpus.",
    )
    cli.add_argument("--query", required=True, help="Question to answer")
    cli.add_argument(
        "--file",
        default=None,
        help="Optional PDF file to auto-ingest before answering.",
    )
    cli.add_argument(
        "--json",
        action="store_true",
        help="Emit answer and metadata as JSON.",
    )
    cli.add_argument(
        "--show-context",
        action="store_true",
        help="Print grounded context before the final answer.",
    )
    return cli.parse_args()


def main() -> None:
    """Run the grounded answer pipeline."""
    args = parse_args()

    index_service = IndexService()

    if args.file:
        index_service.ensure_pdf_indexed(resolve_pdf_path(args.file))

    if not index_service.index.collection_exists():
        raise ValueError(
            "No index exists yet. Provide --file to ingest a document first."
        )

    service = GroundedAnswerService(
        index=index_service.index,
        config=GeneratorConfig(),
    )
    result = service.answer(args.query)

    payload = result.to_dict(include_context=args.show_context)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.show_context and result.context:
        print("#" * 40 + " CONTEXT " + "#" * 40)
        print(result.context)
        print()

    print("#" * 40 + " ANSWER " + "#" * 40)
    print(result.answer)
    print()

    if result.context.sources:
        print(render_sources(result.context.sources))
        print()

    print(
        "timings: "
        f"retrieval={result.timings.retrieval_seconds:.3f}s, "
        f"generation={result.timings.generation_seconds:.3f}s, "
        f"pipeline={result.timings.total_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()