from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer


@dataclass(slots=True)
class ParserConfig:
    max_chunk_tokens: int = 220
    min_chunk_tokens: int = 80
    chunk_tokenizer_model: str = "BAAI/bge-base-en-v1.5"

    allowed_formats: list[InputFormat] = field(
        default_factory=lambda: [
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.MD,
            InputFormat.HTML,
            InputFormat.IMAGE,
        ]
    )

    enable_picture_description: bool = True
    include_picture_chunks: bool = True
    picture_description_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    picture_description_prompt: str = (
        "Describe this picture in three to five sentences. Be precise and concise."
    )
    picture_image_scale: float = 2.0

    extra_format_options: dict[InputFormat, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be greater than 0")
        if self.min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be greater than 0")
        if self.min_chunk_tokens >= self.max_chunk_tokens:
            raise ValueError("min_chunk_tokens must be smaller than max_chunk_tokens")
        if self.picture_image_scale <= 0:
            raise ValueError("picture_image_scale must be greater than 0")

    def build_tokenizer(self) -> HuggingFaceTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.chunk_tokenizer_model)
        return HuggingFaceTokenizer(
            tokenizer=tokenizer,
            max_tokens=self.max_chunk_tokens,
        )

    def build_converter(self) -> DocumentConverter:
        format_options = dict(self.extra_format_options)

        for input_format, format_option in self.build_format_options().items():
            format_options[input_format] = format_option

        return DocumentConverter(
            allowed_formats=self.allowed_formats,
            format_options=format_options or None,
        )

    def build_format_options(self) -> dict[InputFormat, Any]:
        options: dict[InputFormat, Any] = {}

        pdf_option = self._build_pdf_format_option()
        if pdf_option is not None and InputFormat.PDF in self.allowed_formats:
            options[InputFormat.PDF] = pdf_option

        docx_option = self._build_docx_format_option()
        if docx_option is not None and InputFormat.DOCX in self.allowed_formats:
            options[InputFormat.DOCX] = docx_option

        markdown_option = self._build_markdown_format_option()
        if markdown_option is not None and InputFormat.MD in self.allowed_formats:
            options[InputFormat.MD] = markdown_option

        html_option = self._build_html_format_option()
        if html_option is not None and InputFormat.HTML in self.allowed_formats:
            options[InputFormat.HTML] = html_option

        image_option = self._build_image_format_option()
        if image_option is not None and InputFormat.IMAGE in self.allowed_formats:
            options[InputFormat.IMAGE] = image_option

        return options

    def build_text_converter(self) -> DocumentConverter:
        return DocumentConverter(
            allowed_formats=self.allowed_formats,
        )

    def _build_pdf_format_option(self) -> PdfFormatOption | None:
        if not self.enable_picture_description:
            return None

        pipeline_options = PdfPipelineOptions(
            do_picture_description=True,
            picture_description_options=PictureDescriptionVlmOptions(
                repo_id=self.picture_description_model,
                prompt=self.picture_description_prompt,
            ),
            generate_picture_images=True,
            images_scale=self.picture_image_scale,
        )

        return PdfFormatOption(pipeline_options=pipeline_options)

    def _build_docx_format_option(self) -> Any | None:
        return None

    def _build_markdown_format_option(self) -> Any | None:
        return None

    def _build_html_format_option(self) -> Any | None:
        return None

    def _build_image_format_option(self) -> Any | None:
        return None