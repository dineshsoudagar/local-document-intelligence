"""Configuration for grounded answer generation over retrieved local chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config.model_catalog import ModelCatalog, PipelineModels, default_pipeline_models


def _default_project_root() -> Path:
    """Return the repository root based on the config module location."""
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class GeneratorConfig:
    """Runtime settings for grounded answer generation."""

    project_root: Path = field(default_factory=_default_project_root)
    model_catalog: ModelCatalog = field(default_factory=ModelCatalog)
    pipeline_models: PipelineModels = field(default_factory=default_pipeline_models)

    max_context_tokens: int = 6000
    max_chunk_tokens: int = 900
    max_new_tokens: int = 384

    temperature: float = 0.7   
    top_p: float = 0.8
    repetition_penalty: float = 1.05

    default_reasoning_mode: str = "no_think"

    thinking_max_new_tokens: int = 1024
    thinking_temperature: float = 0.6
    thinking_top_p: float = 0.95
    
    system_prompt: str = (
        "You answer questions only from the provided context. "
        "If the context is insufficient, say that the answer is not supported by the retrieved evidence. "
        "Cite supporting evidence inline with bracketed source ids like [1] or [2]. "
        "Do not invent facts, page numbers, or source ids."
    )
    answer_instruction: str = "Write a concise answer grounded in the retrieved context."

    assistant_system_prompt: str = (
        "You are a helpful local assistant. "
        "Answer clearly and directly. "
        "Be honest about uncertainty."
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
        return self.model_catalog.resolve_hf_path(
            self.pipeline_models.generator_key,
            self.project_root,
        )

    def resolve_reasoning_mode(self, reasoning_mode: str | None) -> str:
        """Normalize and validate reasoning mode."""
        mode = (reasoning_mode or self.default_reasoning_mode).strip().lower()
        if mode not in {"think", "no_think"}:
            raise ValueError("reasoning_mode must be one of: think, no_think")
        return mode

    def thinking_enabled(self, reasoning_mode: str | None = None) -> bool:
        """Return whether Qwen thinking mode should be enabled."""
        return self.resolve_reasoning_mode(reasoning_mode) == "think"

    def max_new_tokens_for(self, reasoning_mode: str | None = None) -> int:
        """Return the output token budget for the selected reasoning mode."""
        if self.thinking_enabled(reasoning_mode):
            return self.thinking_max_new_tokens
        return self.max_new_tokens

    def temperature_for(self, reasoning_mode: str | None = None) -> float:
        """Return the sampling temperature for the selected reasoning mode."""
        if self.thinking_enabled(reasoning_mode):
            return self.thinking_temperature
        return self.temperature

    def top_p_for(self, reasoning_mode: str | None = None) -> float:
        """Return the nucleus sampling value for the selected reasoning mode."""
        if self.thinking_enabled(reasoning_mode):
            return self.thinking_top_p
        return self.top_p
    
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

        _ = self.model_catalog.get_hf_model(self.pipeline_models.generator_key)
        model_path = self.generator_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Generator model path does not exist: {model_path}")
