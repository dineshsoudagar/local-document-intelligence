from __future__ import annotations

from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser


def build_parser(name: str, config: ParserConfig) -> DoclingParser:
    if name == "docling":
        return DoclingParser(config)

    if name == "pymupdf":
        raise NotImplementedError(
            "PyMuPDF is disabled until it is migrated to the token-based ParserConfig."
        )

    raise ValueError(f"Unsupported parser: {name}")