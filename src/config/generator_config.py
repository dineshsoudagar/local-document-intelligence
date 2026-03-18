from __future__ import annotations

"""Configuration for grounded answer generation over retrieved local chunks."""

from dataclasses import dataclass, field
from pathlib import Path

from src.config.model_catalog import ModelCatalog


def _default_project_root() -> Path:
    """Return the repository root based on the config module location."""
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class GeneratorConfig:
    """Runtime settings for grounded answer generation."""

    project_root: Path = field(default_factory=_default_project_root)
    model_catalog: ModelCatalog = field(default_factory=ModelCatalog)

    generator_model_key: str = "generator"
    generator_model_path_override: str | Path | None = None

    max_context_tokens: int = 6000
    max_chunk_tokens: int = 900
    max_new_tokens: int = 384

    temperature: float = 0.7
    top_p: float = 0.8
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

    assistant_system_prompt: str = (
        "You are a helpful local assistant. "
        "Answer clearly and directly. "
        "Be honest about uncertainty. "
        "Do not output chain-of-thought or <think> tags. "
        "Return only the final answer."
    )
    assistant_instruction: str = "Answer the user's request directly."

    auto_min_top_rerank_score: float = 0.28
    auto_min_second_rerank_score: float = 0.18

    def __post_init__(self) -> None:
        """Normalize the project root once during initialization."""
        self.project_root = Path(self.project_root).resolve()

    @property
    def generator_model_path(self) -> Path:
        """Return the resolved local generator directory."""
        if self.generator_model_path_override is not None:
            return Path(self.generator_model_path_override).resolve()
        entry = self.model_catalog.get_hf_model(self.generator_model_key)
        return entry.resolve_dir(self.project_root, self.model_catalog.models_root)

    def validate(self) -> None:
        """Validate generation settings and fail early on broken local paths."""
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
        if not 0.0 <= self.auto_min_top_rerank_score <= 1.0:
            raise ValueError("auto_min_top_rerank_score must be in the range [0, 1]")
        if not 0.0 <= self.auto_min_second_rerank_score <= 1.0:
            raise ValueError("auto_min_second_rerank_score must be in the range [0, 1]")
        if self.auto_min_second_rerank_score > self.auto_min_top_rerank_score:
            raise ValueError(
                "auto_min_second_rerank_score must be smaller than or equal to "
                "auto_min_top_rerank_score"
            )
        if self.max_chunk_tokens > self.max_context_tokens:
            raise ValueError("max_chunk_tokens must be smaller than or equal to max_context_tokens")

        _ = self.model_catalog.get_hf_model(self.generator_model_key)
        model_path = self.generator_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Generator model path does not exist: {model_path}")
