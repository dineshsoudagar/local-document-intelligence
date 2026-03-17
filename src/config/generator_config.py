from __future__ import annotations

"""Configuration for local grounded answer generation."""

from dataclasses import dataclass, field
from pathlib import Path

from src.config.model_catalog import ModelCatalog


def _default_project_root() -> Path:
    """Return the repository root based on the config file location."""
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class GeneratorConfig:
    """Configuration for the local generator model and answer budgets."""

    project_root: Path = field(default_factory=_default_project_root)
    model_catalog: ModelCatalog = field(default_factory=ModelCatalog)

    generator_model_name_override: str | Path | None = None

    max_context_tokens: int = 6000
    max_chunk_tokens: int = 900
    max_new_tokens: int = 384
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.05

    system_prompt: str = (
        "You answer questions only from the provided context. "
        "If the context is insufficient, say that the answer is not supported by the retrieved evidence. "
        "Cite supporting evidence inline with bracketed source ids like [1] or [2]. "
        "Do not invent facts, page numbers, or source ids. "
        "Do not output chain-of-thought or <think> tags. "
        "Return only the final answer."
    )
    answer_instruction: str = "Write a concise answer grounded in the retrieved context."

    def __post_init__(self) -> None:
        """Normalize the project root once during initialization."""
        self.project_root = Path(self.project_root).resolve()

    @property
    def generator_model_name(self) -> str:
        """Return the resolved local generator model directory."""
        if self.generator_model_name_override is not None:
            return str(Path(self.generator_model_name_override).resolve())
        return str(self.model_catalog.generator_path(self.project_root))

    def validate(self) -> None:
        """Validate answer-generation settings and verify local model availability."""
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than 0")
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be greater than 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("top_p must be in the range (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0")

        generator_path = Path(self.generator_model_name)
        if not generator_path.exists():
            raise FileNotFoundError(
                f"Generator model path does not exist: {generator_path}"
            )
