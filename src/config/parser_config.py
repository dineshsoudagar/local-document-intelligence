from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from src.config.model_catalog import ModelCatalog


def _default_project_root() -> Path:
    """Resolve the repository root from the config module location."""
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class ParserConfig:
    """Configuration for Docling parsing, chunk tokenization, and offline model paths."""

    project_root: Path = field(default_factory=_default_project_root)
    model_catalog: ModelCatalog = field(default_factory=ModelCatalog)

    chunk_tokenizer_model_override: str | None = None
    docling_artifacts_path_override: str | None = None
    picture_description_model_override: str | None = None

    max_chunk_tokens: int = 260
    min_chunk_tokens: int = 80

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
    picture_description_prompt: str = (
        "Describe this picture in three to five sentences. Be precise and concise."
    )
    picture_image_scale: float = 2.0

    extra_format_options: dict[InputFormat, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()

    @property
    def chunk_tokenizer_model(self) -> str:
        """Return the local tokenizer directory used for chunk token counting."""
        if self.chunk_tokenizer_model_override:
            return str(Path(self.chunk_tokenizer_model_override).resolve())
        return str(self.model_catalog.chunk_tokenizer_path(self.project_root))

    @property
    def docling_artifacts_path(self) -> str:
        """Return the local Docling artifact directory."""
        if self.docling_artifacts_path_override:
            return str(Path(self.docling_artifacts_path_override).resolve())
        return str(self.model_catalog.docling_artifacts_path(self.project_root))

    @property
    def picture_description_model(self) -> str:
        """Return the local picture description model directory."""
        if self.picture_description_model_override:
            return str(Path(self.picture_description_model_override).resolve())
        return str(self.model_catalog.picture_description_path(self.project_root))

    def validate(self) -> None:
        """Validate numeric settings and verify that local offline assets exist."""
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be greater than 0")
        if self.min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be greater than 0")
        if self.min_chunk_tokens >= self.max_chunk_tokens:
            raise ValueError("min_chunk_tokens must be smaller than max_chunk_tokens")
        if self.picture_image_scale <= 0:
            raise ValueError("picture_image_scale must be greater than 0")

        tokenizer_path = Path(self.chunk_tokenizer_model)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Chunk tokenizer path does not exist: {tokenizer_path}"
            )

        artifacts_path = Path(self.docling_artifacts_path)
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"Docling artifacts path does not exist: {artifacts_path}"
            )

        if self.enable_picture_description:
            picture_model_path = Path(self.picture_description_model)
            if not picture_model_path.exists():
                raise FileNotFoundError(
                    "Picture description model path does not exist: "
                    f"{picture_model_path}"
                )

    def build_tokenizer(self) -> HuggingFaceTokenizer:
        """Build the chunk tokenizer from local files only."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.chunk_tokenizer_model,
            local_files_only=True,
        )
        return HuggingFaceTokenizer(
            tokenizer=tokenizer,
            max_tokens=self.max_chunk_tokens,
        )

    def build_converter(self) -> DocumentConverter:
        """Build the main converter with picture description enabled when configured."""
        format_options = dict(self.extra_format_options)
        format_options.update(
            self.build_format_options(
                enable_picture_description=self.enable_picture_description
            )
        )
        return DocumentConverter(
            allowed_formats=self.allowed_formats,
            format_options=format_options or None,
        )

    def build_text_converter(self) -> DocumentConverter:
        """Build a text-focused converter that still uses local Docling artifacts."""
        format_options = dict(self.extra_format_options)
        format_options.update(self.build_format_options(enable_picture_description=False))
        return DocumentConverter(
            allowed_formats=self.allowed_formats,
            format_options=format_options or None,
        )

    def build_format_options(
        self,
        enable_picture_description: bool | None = None,
    ) -> dict[InputFormat, Any]:
        """Build format-specific converter options for the enabled input types."""
        use_picture_description = (
            self.enable_picture_description
            if enable_picture_description is None
            else enable_picture_description
        )

        options: dict[InputFormat, Any] = {}

        pdf_option = self._build_pdf_format_option(
            enable_picture_description=use_picture_description
        )
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

        image_option = self._build_image_format_option(
            enable_picture_description=use_picture_description
        )
        if image_option is not None and InputFormat.IMAGE in self.allowed_formats:
            options[InputFormat.IMAGE] = image_option

        return options

    def _build_pdf_format_option(
        self,
        enable_picture_description: bool,
    ) -> PdfFormatOption:
        """Build the PDF pipeline option wired to local Docling artifacts."""
        pipeline_options = self._build_pdf_pipeline_options(
            enable_picture_description=enable_picture_description
        )
        return PdfFormatOption(pipeline_options=pipeline_options)

    def _build_docx_format_option(self) -> Any | None:
        return None

    def _build_markdown_format_option(self) -> Any | None:
        return None

    def _build_html_format_option(self) -> Any | None:
        return None

    def _build_image_format_option(
        self,
        enable_picture_description: bool,
    ) -> Any | None:
        return None

    def _build_pdf_pipeline_options(
        self,
        enable_picture_description: bool,
    ) -> PdfPipelineOptions:
        """Build PDF pipeline options for fully local Docling execution."""
        pipeline_options = PdfPipelineOptions(
            artifacts_path=self.docling_artifacts_path,
        )

        if enable_picture_description:
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                repo_id=self.picture_description_model,
                prompt=self.picture_description_prompt,
            )
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = self.picture_image_scale
        else:
            pipeline_options.do_picture_description = False
            pipeline_options.generate_picture_images = False

        return pipeline_options
