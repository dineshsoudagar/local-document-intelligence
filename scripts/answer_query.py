from __future__ import annotations

"""CLI entrypoint for grounded answer generation over retrieved local chunks."""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from docling.datamodel.base_models import InputFormat
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.index_config import IndexConfig
from src.config.model_catalog import ModelCatalog
from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser
from src.retrieval.qdrant_hybrid_index import QdrantHybridIndex, RetrievedChunk


@dataclass(slots=True)
class AnswerSource:
    """Serializable source reference attached to a generated answer."""

    rank: int
    chunk_id: str
    source_file: str | None
    page_start: int | None
    page_end: int | None
    rerank_score: float | None
    fusion_score: float | None
    headings: list[str] | None
    block_type: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "rerank_score": self.rerank_score,
            "fusion_score": self.fusion_score,
            "headings": self.headings,
            "block_type": self.block_type,
        }


class LocalQwenGenerator:
    """Minimal local wrapper for grounded text generation with Qwen instruct models."""

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            local_files_only=True,
            padding_side="left",
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if self._device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **model_kwargs,
        ).eval()

        if self._device.type != "cuda":
            self._model.to(self._device)

    def count_tokens(self, text: str) -> int:
        """Return the token count for a text fragment."""
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate a text fragment to a maximum token budget."""
        if max_tokens <= 0:
            return ""
        token_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_tokens:
            return text
        return self._tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True).strip()

    def generate(
        self,
        *,
        query: str,
        context: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        """Generate a grounded answer from the supplied context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions only from the provided context. "
                    "If the context is insufficient, say that the answer is not supported by the retrieved evidence. "
                    "Cite supporting evidence inline with bracketed source ids like [1] or [2]. "
                    "Do not invent facts, page numbers, or source ids. "
                    "Do not output chain-of-thought or <think> tags. "
                    "Return only the final answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Write a concise answer grounded in the retrieved context."
                ),
            },
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "System:\n"
                f"{messages[0]['content']}\n\n"
                "User:\n"
                f"{messages[1]['content']}\n\n"
                "Assistant:\n"
            )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._strip_reasoning(answer)

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove visible reasoning tags if the model emits them."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()


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


def ensure_index(args: argparse.Namespace) -> QdrantHybridIndex:
    """Return a ready-to-query index, building it from PDF input when needed."""
    index = QdrantHybridIndex(IndexConfig())
    if args.rebuild or not index.collection_exists():
        if not args.pdf:
            raise ValueError(
                "--pdf is required when the index does not exist or when --rebuild is used"
            )
        parser_config = ParserConfig(
            allowed_formats=[InputFormat.PDF],
            enable_picture_description=False,
            include_picture_chunks=True,
        )
        parser = DoclingParser(parser_config)
        chunks = parser.parse(args.pdf, doc_id=build_doc_id(args.pdf))
        index.build(chunks=chunks, rebuild=args.rebuild)
    return index


def build_context(
    generator: LocalQwenGenerator,
    chunks: Sequence[RetrievedChunk],
    *,
    max_context_tokens: int,
    max_chunk_tokens: int,
) -> tuple[str, list[AnswerSource]]:
    """Build a prompt context from retrieved chunks within a token budget."""
    blocks: list[str] = []
    sources: list[AnswerSource] = []
    used_tokens = 0

    for rank, chunk in enumerate(chunks, start=1):
        text = generator.truncate_text(chunk.text.strip(), max_chunk_tokens)
        if not text:
            continue

        headings = chunk.metadata.get("headings")
        heading_text = " > ".join(headings) if isinstance(headings, list) and headings else "-"
        block_type = str(chunk.metadata.get("block_type") or "text")
        source_file = chunk.source_file or "unknown"
        pages = format_pages(chunk.page_start, chunk.page_end)

        block = (
            f"[{rank}] source_file: {source_file}\n"
            f"[{rank}] pages: {pages}\n"
            f"[{rank}] block_type: {block_type}\n"
            f"[{rank}] headings: {heading_text}\n"
            f"[{rank}] content:\n{text}"
        )
        block_tokens = generator.count_tokens(block)

        if blocks and used_tokens + block_tokens > max_context_tokens:
            break

        if not blocks and block_tokens > max_context_tokens:
            shrink_target = max(128, max_context_tokens // 2)
            trimmed_text = generator.truncate_text(text, shrink_target)
            block = (
                f"[{rank}] source_file: {source_file}\n"
                f"[{rank}] pages: {pages}\n"
                f"[{rank}] block_type: {block_type}\n"
                f"[{rank}] headings: {heading_text}\n"
                f"[{rank}] content:\n{trimmed_text}"
            )
            block_tokens = generator.count_tokens(block)

        if block_tokens > max_context_tokens:
            continue

        blocks.append(block)
        used_tokens += block_tokens
        sources.append(
            AnswerSource(
                rank=rank,
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                rerank_score=chunk.rerank_score,
                fusion_score=chunk.fused_score,
                headings=headings if isinstance(headings, list) else None,
                block_type=str(chunk.metadata.get("block_type")) if chunk.metadata.get("block_type") is not None else None,
            )
        )

    return "\n\n".join(blocks).strip(), sources


def format_pages(page_start: int | None, page_end: int | None) -> str:
    """Return a stable printable page span."""
    if page_start is None and page_end is None:
        return "unknown"
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"


def render_sources(sources: Sequence[AnswerSource]) -> str:
    """Render a readable source appendix."""
    lines = ["Sources:"]
    for source in sources:
        heading_text = " > ".join(source.headings) if source.headings else "-"
        lines.append(
            "  "
            f"[{source.rank}] {source.source_file or 'unknown'} | "
            f"pages {format_pages(source.page_start, source.page_end)} | "
            f"chunk_id={source.chunk_id} | "
            f"block_type={source.block_type or '-'} | "
            f"headings={heading_text} | "
            f"rerank={format_score(source.rerank_score)} | "
            f"fusion={format_score(source.fusion_score)}"
        )
    return "\n".join(lines)


def format_score(value: float | None) -> str:
    """Format a numeric score for display."""
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    """Retrieve grounded chunks and generate a final answer."""
    args = parse_args()

    if args.retrieval_top_k <= 0:
        raise ValueError("--retrieval-top-k must be greater than 0")
    if args.max_context_tokens <= 0:
        raise ValueError("--max-context-tokens must be greater than 0")
    if args.max_chunk_tokens <= 0:
        raise ValueError("--max-chunk-tokens must be greater than 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than 0")
    if args.top_p <= 0 or args.top_p > 1:
        raise ValueError("--top-p must be in the range (0, 1]")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be greater than 0")

    catalog = ModelCatalog()
    project_root = Path(__file__).resolve().parents[1]
    generator_path = catalog.generator_path(project_root)
    if not generator_path.exists():
        raise FileNotFoundError(f"Generator model path does not exist: {generator_path}")

    index_start = time.perf_counter()
    index = ensure_index(args)
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

    generator = LocalQwenGenerator(generator_path)
    context, sources = build_context(
        generator,
        retrieved_chunks,
        max_context_tokens=args.max_context_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
    )

    if not context:
        raise RuntimeError("Retrieved chunks were available, but no context could fit the prompt budget")

    generation_start = time.perf_counter()
    answer = generator.generate(
        query=args.query,
        context=context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    generation_seconds = time.perf_counter() - generation_start
    total_seconds = time.perf_counter() - index_start

    payload = {
        "query": args.query,
        "answer": answer,
        "sources": [source.to_dict() for source in sources],
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
        print(context)
        print()

    print("#" * 40 + " ANSWER " + "#" * 40)
    print(answer)
    print()
    print(render_sources(sources))
    print()
    print(
        "timings: "
        f"retrieval={retrieval_seconds:.3f}s, "
        f"generation={generation_seconds:.3f}s, "
        f"total={total_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
