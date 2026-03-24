"""CLI entrypoint for answer generation over the local corpus."""

from __future__ import annotations

import argparse
import time

from src.config.generator_config import GeneratorConfig
from src.generation.answer_service import GroundedAnswerService
from src.generation.context_builder import GroundedContext, render_sources
from src.indexing.index_service import IndexService
from src.utils.io import resolve_pdf_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for answer generation."""
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
        "--mode",
        choices=["grounded", "chat", "auto"],
        default="grounded",
        help=(
            "Answer mode. 'grounded' always uses retrieval, "
            "'chat' never uses retrieval, "
            "'auto' switches by rerank confidence."
        ),
    )
    cli.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated text to stdout.",
    )
    return cli.parse_args()


def _print_sources(context: GroundedContext) -> None:
    """Print rendered evidence sources when available."""
    if not context.sources:
        return

    print(render_sources(context.sources))
    print()


def _print_timings(
        *,
        retrieval_seconds: float,
        generation_seconds: float,
        total_seconds: float,
) -> None:
    """Print pipeline timings."""
    print(
        "timings: "
        f"retrieval={retrieval_seconds:.3f}s, "
        f"generation={generation_seconds:.3f}s, "
        f"pipeline={total_seconds:.3f}s"
    )


def main() -> None:
    """Run the answer pipeline."""
    args = parse_args()

    index_service = IndexService()
    target_doc_ids: list[str] | None = None

    if args.file:
        doc_id = index_service.ensure_pdf_indexed(resolve_pdf_path(args.file))
        target_doc_ids = [doc_id]

    if args.mode == "grounded" and not index_service.index.collection_exists():
        raise ValueError(
            "No index exists yet. Provide --file to ingest a document first, or use --mode chat."
        )

    service = GroundedAnswerService(
        index=index_service.index,
        config=GeneratorConfig(),
    )

    if args.stream:
        start_payload, text_stream = service.stream(args.query, mode=args.mode, doc_ids=target_doc_ids)

        print("#" * 40 + " ANSWER " + "#" * 40)

        generation_started_at = time.perf_counter()
        for token_text in text_stream:
            if not token_text:
                continue
            print(token_text, end="", flush=True)

        generation_seconds = time.perf_counter() - generation_started_at

        print()
        print()

        stream_context = GroundedContext(
            text="",
            sources=start_payload.sources,
            used_tokens=start_payload.used_context_tokens,
        )
        _print_sources(stream_context)

        _print_timings(
            retrieval_seconds=start_payload.retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=start_payload.retrieval_seconds + generation_seconds,
        )
        return

    result = service.answer(
        args.query,
        mode=args.mode,
        doc_ids=target_doc_ids,
    )

    print("#" * 40 + " ANSWER " + "#" * 40)
    print(result.answer)
    print()

    _print_sources(result.context)

    _print_timings(
        retrieval_seconds=result.timings.retrieval_seconds,
        generation_seconds=result.timings.generation_seconds,
        total_seconds=result.timings.total_seconds,
    )


if __name__ == "__main__":
    main()
