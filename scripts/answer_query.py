from __future__ import annotations

"""CLI entrypoint for grounded answer generation over retrieved local chunks."""

import argparse
import json
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat

from src.config.generator_config import GeneratorConfig
from src.config.index_config import IndexConfig
from src.config.parser_config import ParserConfig
from src.generation.context_builder import build_grounded_context, render_sources
from src.parser.docling_parser import DoclingParser
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex
from src.retrieval.qwen_models import LocalQwenGenerator


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for grounded retrieval and answer generation."""
    cli = argparse.ArgumentParser(
        description="Retrieve local chunks and answer a query with a local generator.",
    )
    cli.add_argument("--query", required=True, help="Question to answer")
    cli.add_argument(
        "--pdf",
        default=None,
        help="Optional PDF path. Used only when indexing is needed or rebuild is requested.",
    )
    cli.add_argument("--rebuild", action="store_true", help="Rebuild the index from the PDF")
    cli.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing models and storage directories.",
    )
    cli.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks passed to the generator.",
    )
    cli.add_argument(
        "--max-context-tokens",
        type=int,
        default=6000,
        help="Maximum total token budget reserved for retrieved context.",
    )
    cli.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=900,
        help="Maximum token budget per retrieved chunk before trimming.",
    )
    cli.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Maximum generated answer length.",
    )
    cli.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic decoding.",
    )
    cli.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling threshold")
    cli.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Penalty applied during generation to reduce repetition.",
    )
    cli.add_argument(
        "--show-context",
        action="store_true",
        help="Print the grounded context blocks before the answer.",
    )
    cli.add_argument(
        "--json",
        action="store_true",
        help="Emit answer, sources, and timings as JSON.",
    )
    return cli.parse_args()


def build_doc_id(path: str) -> str:
    """Convert a file path into a stable document identifier."""
    pdf_path = Path(path)
    return pdf_path.stem.strip().replace(" ", "_").replace("-", "_").lower()


def build_index_config(project_root: Path) -> IndexConfig:
    """Build the retrieval config for the current project root."""
    config = IndexConfig(project_root=project_root)
    config.validate()
    return config


def build_generator_config(args: argparse.Namespace, project_root: Path) -> GeneratorConfig:
    """Build the generation config from CLI inputs."""
    config = GeneratorConfig(
        project_root=project_root,
        max_context_tokens=args.max_context_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    config.validate()
    return config


def ensure_index(args: argparse.Namespace, index_config: IndexConfig) -> QdrantHybridIndex:
    """Return a ready-to-query index, building it from PDF input when needed."""
    index = QdrantHybridIndex(index_config)
    if args.rebuild or not index.collection_exists():
        if not args.pdf:
            raise ValueError(
                "--pdf is required when the index does not exist or when --rebuild is used"
            )
        parser_config = ParserConfig(
            project_root=index_config.project_root,
            allowed_formats=[InputFormat.PDF],
            enable_picture_description=False,
            include_picture_chunks=True,
        )
        parser = DoclingParser(parser_config)
        chunks = parser.parse(args.pdf, doc_id=build_doc_id(args.pdf))
        index.build(chunks=chunks, rebuild=args.rebuild)
    return index


def main() -> None:
    """Retrieve grounded chunks and generate a final answer."""
    args = parse_args()

    if args.retrieval_top_k <= 0:
        raise ValueError("--retrieval-top-k must be greater than 0")

    project_root = args.project_root.resolve()
    index_config = build_index_config(project_root)
    generator_config = build_generator_config(args, project_root)

    index_start = time.perf_counter()
    index = ensure_index(args, index_config)
    retrieval_start = time.perf_counter()
    retrieved_chunks = index.search(args.query, top_k=args.retrieval_top_k)
    retrieval_seconds = time.perf_counter() - retrieval_start

    if not retrieved_chunks:
        payload = {
            "query": args.query,
            "answer": "No relevant chunks were retrieved.",
            "sources": [],
            "timings": {
                "index_setup_seconds": time.perf_counter() - index_start,
                "retrieval_seconds": retrieval_seconds,
                "generation_seconds": 0.0,
            },
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(payload["answer"])
        return

    generator = LocalQwenGenerator(generator_config.generator_model_name)
    grounded_context = build_grounded_context(
        generator,
        retrieved_chunks,
        max_context_tokens=generator_config.max_context_tokens,
        max_chunk_tokens=generator_config.max_chunk_tokens,
    )

    if not grounded_context.text:
        raise RuntimeError(
            "Retrieved chunks were available, but no context could fit the prompt budget"
        )

    generation_start = time.perf_counter()
    answer = generator.generate_grounded_answer(
        query=args.query,
        context=grounded_context.text,
        system_prompt=generator_config.system_prompt,
        answer_instruction=generator_config.answer_instruction,
        max_new_tokens=generator_config.max_new_tokens,
        temperature=generator_config.temperature,
        top_p=generator_config.top_p,
        repetition_penalty=generator_config.repetition_penalty,
    )
    generation_seconds = time.perf_counter() - generation_start
    total_seconds = time.perf_counter() - index_start

    payload = {
        "query": args.query,
        "answer": answer,
        "sources": [source.to_dict() for source in grounded_context.sources],
        "timings": {
            "total_seconds": total_seconds,
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.show_context:
        print("#" * 40 + " CONTEXT " + "#" * 40)
        print(grounded_context.text)
        print()

    print("#" * 40 + " ANSWER " + "#" * 40)
    print(answer)
    print()
    print(render_sources(grounded_context.sources))
    print()
    print(
        "timings: "
        f"retrieval={retrieval_seconds:.3f}s, "
        f"generation={generation_seconds:.3f}s, "
        f"total={total_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
