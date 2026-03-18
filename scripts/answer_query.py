from __future__ import annotations

from pathlib import Path

from src.utils.io import resolve_pdf_path

"""CLI entrypoint for grounded answer generation over the local corpus."""

import argparse
import json
import time
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

    cli.add_argument(
        "--mode",
        choices=["grounded", "chat", "auto"],
        default="grounded",
        help="Answer mode. 'grounded' always uses retrieval, 'chat' never uses retrieval, 'auto' switches by rerank "
             "confidence.",
    )
    cli.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated text to stdout.",
    )
    return cli.parse_args()


def main() -> None:
    """Run the grounded answer pipeline."""
    args = parse_args()

    index_service = IndexService()

    if args.file:
        index_service.ensure_pdf_indexed(resolve_pdf_path(args.file))

    if args.mode == "grounded" and not index_service.index.collection_exists():
        raise ValueError(
            "No index exists yet. Provide --file to ingest a document first, or use --mode chat."
        )

    service = GroundedAnswerService(
        index=index_service.index,
        config=GeneratorConfig(),
    )


    if args.json and args.stream:
        raise ValueError("--json and --stream cannot be used together.")

    if args.stream:
        context, _, retrieval_seconds, stream = service.stream_answer(args.query)

        if args.show_context and context:
            print("#" * 40 + " CONTEXT " + "#" * 40)
            print(context)
            print()

        print("#" * 40 + " ANSWER " + "#" * 40)

        generation_started_at = time.perf_counter()
        for chunk in stream:
            print(chunk, end="", flush=True)
        generation_seconds = time.perf_counter() - generation_started_at

        print()
        print()

        if context.sources:
            print(render_sources(context.sources))
            print()

        print(
            "timings: "
            f"retrieval={retrieval_seconds:.3f}s, "
            f"generation={generation_seconds:.3f}s, "
            f"pipeline={retrieval_seconds + generation_seconds:.3f}s"
        )
        return

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