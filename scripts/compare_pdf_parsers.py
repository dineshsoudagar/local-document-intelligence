import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def compute_text_metrics(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = re.findall(r"\b\w+\b", text)
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    total_chars = len(text)
    non_alpha_ratio = 0.0 if total_chars == 0 else 1.0 - (alpha_chars / total_chars)

    line_word_counts = [len(re.findall(r"\b\w+\b", line)) for line in lines]
    short_lines = sum(1 for count in line_word_counts if 0 < count <= 3)
    short_line_ratio = 0.0 if not lines else short_lines / len(lines)

    avg_line_words = 0.0 if not line_word_counts else statistics.mean(line_word_counts)
    avg_word_len = 0.0 if not words else statistics.mean(len(word) for word in words)

    suspicious_lines = 0
    for line in lines:
        tokens = re.findall(r"\b\w+\b", line)
        if not tokens:
            continue
        uppercase_ratio = sum(1 for token in tokens if token.isupper()) / len(tokens)
        punctuation_ratio = (
            sum(1 for ch in line if not ch.isalnum() and not ch.isspace()) / max(len(line), 1)
        )
        if len(tokens) <= 4 and (uppercase_ratio > 0.5 or punctuation_ratio > 0.2):
            suspicious_lines += 1

    suspicious_line_ratio = 0.0 if not lines else suspicious_lines / len(lines)

    return {
        "char_count": total_chars,
        "word_count": len(words),
        "line_count": len(lines),
        "avg_line_words": round(avg_line_words, 2),
        "avg_word_len": round(avg_word_len, 2),
        "non_alpha_ratio": round(non_alpha_ratio, 4),
        "short_line_ratio": round(short_line_ratio, 4),
        "suspicious_line_ratio": round(suspicious_line_ratio, 4),
    }


def split_markdown_into_blocks(text: str) -> list[dict[str, Any]]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    blocks: list[dict[str, Any]] = []

    for index, part in enumerate(parts):
        block_type = "paragraph"
        if part.startswith("#"):
            block_type = "heading"
        elif re.match(r"^[-*]\s+", part):
            block_type = "list"
        elif re.match(r"^\|.+\|$", part.splitlines()[0]):
            block_type = "table"
        print(part)
        blocks.append(
            {
                "block_index": index,
                "block_type": block_type,
                "text": part,
            }
        )

    return blocks


def run_current_parser(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    from src.config.parser_config import ParserConfig
    from src.parser.pdf_parser import PDFParser

    parser = PDFParser(
        config=ParserConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
    chunks = parser.parse(pdf_path)
    text = "\n\n".join(chunk.text for chunk in chunks)

    blocks = [
        {
            "block_index": chunk.chunk_index,
            "block_type": "chunk",
            "chunk_id": chunk.chunk_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "text": chunk.text,
        }
        for chunk in chunks
    ]

    metrics = compute_text_metrics(text)
    metrics["block_count"] = len(blocks)

    return {
        "parser": "current_parser",
        "text": text,
        "blocks": blocks,
        "metrics": metrics,
    }


def run_pymupdf4llm_parser(pdf_path: Path) -> dict[str, Any]:
    import pymupdf4llm

    text = normalize_text(pymupdf4llm.to_markdown(str(pdf_path)))
    blocks = split_markdown_into_blocks(text)

    metrics = compute_text_metrics(text)
    metrics["block_count"] = len(blocks)

    return {
        "parser": "pymupdf4llm",
        "text": text,
        "blocks": blocks,
        "metrics": metrics,
    }


def run_docling_parser(pdf_path: Path) -> dict[str, Any]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    document = result.document
    text = normalize_text(document.export_to_markdown())
    blocks = split_markdown_into_blocks(text)

    metrics = compute_text_metrics(text)
    metrics["block_count"] = len(blocks)

    return {
        "parser": "docling",
        "text": text,
        "blocks": blocks,
        "metrics": metrics,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(outputs: list[dict[str, Any]]) -> None:
    ordered_keys = [
        "char_count",
        "word_count",
        "line_count",
        "avg_line_words",
        "avg_word_len",
        "non_alpha_ratio",
        "short_line_ratio",
        "suspicious_line_ratio",
        "block_count",
    ]

    for output in outputs:
        print("=" * 100)
        print(output["parser"])
        for key in ordered_keys:
            if key in output["metrics"]:
                print(f"{key}: {output['metrics'][key]}")
        print("preview:")
        print(output["text"][:1200])
        print()


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--pdf", required=True)
    cli.add_argument("--out-dir", required=True)
    cli.add_argument("--chunk-size", type=int, default=512)
    cli.add_argument("--chunk-overlap", type=int, default=64)
    cli.add_argument(
        "--parsers",
        nargs="+",
        default=["current", "pymupdf4llm", "docling"],
        choices=["current", "pymupdf4llm", "docling"],
    )

    args = cli.parse_args()

    pdf_path = Path(args.pdf).resolve()
    out_dir = Path(args.out_dir).resolve()

    outputs: list[dict[str, Any]] = []

    if "current" in args.parsers:
        outputs.append(
            run_current_parser(
                pdf_path=pdf_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )

    if "pymupdf4llm" in args.parsers:
        outputs.append(run_pymupdf4llm_parser(pdf_path=pdf_path))

    if "docling" in args.parsers:
        outputs.append(run_docling_parser(pdf_path=pdf_path))

    summary = {output["parser"]: output["metrics"] for output in outputs}

    for output in outputs:
        parser_dir = out_dir / output["parser"]
        write_text(parser_dir / "text.txt", output["text"])
        write_json(parser_dir / "blocks.json", output["blocks"])
        write_json(parser_dir / "metrics.json", output["metrics"])

    write_json(out_dir / "summary.json", summary)
    print_summary(outputs)


if __name__ == "__main__":
    main()