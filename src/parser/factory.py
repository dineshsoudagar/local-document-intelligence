from src.config.parser_config import ParserConfig
from src.parser.docling_parser import DoclingParser
from src.parser.pdf_parser import PDFParser


def build_parser(name: str, config: ParserConfig):
    if name == "docling":
        return DoclingParser(config)
    if name == "pymupdf":
        return PDFParser(config)
    raise ValueError(f"Unsupported parser: {name}")