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
    embedding_tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    allowed_formats: list[InputFormat] | None = None

    enable_picture_description: bool = False
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
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_tokenizer_model)
        return HuggingFaceTokenizer(
            tokenizer=tokenizer,
            max_tokens=self.max_chunk_tokens,
        )

    def build_converter(self) -> DocumentConverter:
        format_options = dict(self.extra_format_options)

        pdf_option = self._build_pdf_format_option()
        if pdf_option is not None:
            format_options[InputFormat.PDF] = pdf_option

        return DocumentConverter(
            allowed_formats=self.allowed_formats,
            format_options=format_options or None,
        )

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