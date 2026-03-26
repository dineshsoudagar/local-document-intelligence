"""Configuration for grounded answer generation over retrieved local chunks."""

from __future__ import annotations

from pathlib import Path
from src.config.model_catalog import ModelCatalog, PipelineModels, default_pipeline_models
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal

GeneratorLoadMode = Literal["standard", "bnb_8bit", "bnb_4bit"]
GeneratorDType = Literal["auto", "float16", "bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class GeneratorLoadPreset:
    """Named generator loading preset for memory and runtime tradeoffs."""

    key: str
    label: str
    description: str
    memory_hint: str
    generator_load_mode: GeneratorLoadMode
    generator_dtype: GeneratorDType = "bfloat16"
    generator_device_map: str | None = "auto"
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_int8_enable_fp32_cpu_offload: bool = False

    def to_overrides(self) -> dict[str, Any]:
        """Return dataclass field overrides for GeneratorConfig."""
        return {
            "generator_load_mode": self.generator_load_mode,
            "generator_dtype": self.generator_dtype,
            "generator_device_map": self.generator_device_map,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_int8_enable_fp32_cpu_offload": self.bnb_int8_enable_fp32_cpu_offload,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly preset description for UI use."""
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "memory_hint": self.memory_hint,
            "generator_load_mode": self.generator_load_mode,
            "generator_dtype": self.generator_dtype,
            "generator_device_map": self.generator_device_map,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_int8_enable_fp32_cpu_offload": self.bnb_int8_enable_fp32_cpu_offload,
        }


GENERATOR_LOAD_PRESETS: dict[str, GeneratorLoadPreset] = {
    "standard": GeneratorLoadPreset(
        key="standard",
        label="Standard",
        description="Standard loading without bitsandbytes quantization.",
        memory_hint="Highest memory usage. Best baseline compatibility.",
        generator_load_mode="standard",
        generator_dtype="bfloat16",
        generator_device_map="auto",
    ),
    "bnb_8bit": GeneratorLoadPreset(
        key="bnb_8bit",
        label="8-bit (bnb)",
        description="Reduced memory usage with 8-bit loading and better quality retention than 4-bit.",
        memory_hint="Good choice for mid-range GPUs.",
        generator_load_mode="bnb_8bit",
        generator_dtype="bfloat16",
        generator_device_map="auto",
    ),
    "bnb_4bit": GeneratorLoadPreset(
        key="bnb_4bit",
        label="4-bit (bnb)",
        description="Aggressive VRAM reduction using 4-bit NF4 quantization.",
        memory_hint="Best fit for constrained GPUs. Quality may drop slightly.",
        generator_load_mode="bnb_4bit",
        generator_dtype="bfloat16",
        generator_device_map="auto",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
    "cpu_safe": GeneratorLoadPreset(
        key="cpu_safe",
        label="CPU safe",
        description="No bitsandbytes quantization. Keeps loading path compatible with CPU-only runs.",
        memory_hint="Slowest generation. Use when CUDA is unavailable.",
        generator_load_mode="standard",
        generator_dtype="float32",
        generator_device_map=None,
    ),
}

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
    
    generator_load_mode: GeneratorLoadMode = "standard"
    generator_dtype: GeneratorDType = "bfloat16"
    generator_device_map: str | None = "auto"

    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_int8_enable_fp32_cpu_offload: bool = False

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

    @classmethod
    def available_load_presets(cls) -> list[dict[str, Any]]:
        """Return JSON-friendly preset metadata for UI consumers."""
        return [preset.to_dict() for preset in GENERATOR_LOAD_PRESETS.values()]

    @classmethod
    def get_load_preset(cls, preset_key: str) -> GeneratorLoadPreset:
        """Return one named loading preset."""
        try:
            return GENERATOR_LOAD_PRESETS[preset_key]
        except KeyError as exc:
            available = ", ".join(sorted(GENERATOR_LOAD_PRESETS))
            raise ValueError(
                f"Unknown generator load preset: {preset_key}. Available presets: {available}"
            ) from exc

    def with_load_preset(self, preset_key: str) -> "GeneratorConfig":
        """Return a copied config with one named loading preset applied."""
        preset = self.get_load_preset(preset_key)
        return replace(self, **preset.to_overrides())

    def current_load_preset_key(self) -> str | None:
        """Return the matching preset key if the active fields match a known preset."""
        for key, preset in GENERATOR_LOAD_PRESETS.items():
            overrides = preset.to_overrides()
            if all(getattr(self, field_name) == value for field_name, value in overrides.items()):
                return key
        return None

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
        if self.generator_load_mode not in {"standard", "bnb_8bit", "bnb_4bit"}:
            raise ValueError(
                "generator_load_mode must be one of: standard, bnb_8bit, bnb_4bit"
            )
        if self.generator_dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(
                "generator_dtype must be one of: auto, float16, bfloat16, float32"
            )
        _ = self.model_catalog.get_hf_model(self.pipeline_models.generator_key)
        model_path = self.generator_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Generator model path does not exist: {model_path}")
