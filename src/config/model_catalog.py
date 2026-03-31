from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SelectableModelRole = Literal["generator", "embedder", "reranker", "picture_description"]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Definition for a Hugging Face model stored under the project models directory."""

    key: str
    repo_id: str
    relative_dir: str
    revision: str = "main"
    role: SelectableModelRole | None = None
    selectable: bool = False
    label: str | None = None
    description: str | None = None
    size_hint: str | None = None
    vram_hint: str | None = None

    def resolve_dir(self, project_root: str | Path, models_root: str) -> Path:
        """Return the resolved local directory for this model."""
        return Path(project_root).resolve() / models_root / self.relative_dir

    def to_setup_option(self) -> dict[str, str | None]:
        """Return JSON-friendly setup metadata for one selectable model."""
        return {
            "key": self.key,
            "role": self.role,
            "label": self.label or self.key,
            "description": self.description,
            "size_hint": self.size_hint,
            "vram_hint": self.vram_hint,
            "repo_id": self.repo_id,
        }


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """Definition for non-Hugging Face assets stored under the project models directory."""

    key: str
    relative_dir: str
    label: str | None = None
    description: str | None = None

    def resolve_dir(self, project_root: str | Path, models_root: str) -> Path:
        """Return the resolved local directory for this artifact bundle."""
        return Path(project_root).resolve() / models_root / self.relative_dir


@dataclass(frozen=True, slots=True)
class PipelineModels:
    """Selected assets for each pipeline role."""

    embedder_key: str = "qwen3_embedding_0_6b"
    reranker_key: str = "qwen3_reranker_0_6b"
    generator_key: str = "qwen3_4b"
    picture_description_key: str = "smolvlm_256m_instruct"
    docling_artifacts_key: str = "docling_artifacts"
    chunk_tokenizer_key: str | None = None

    def effective_chunk_tokenizer_key(self) -> str:
        """Return the tokenizer asset key used for chunking."""
        return self.chunk_tokenizer_key or self.embedder_key

    def required_asset_keys(self) -> tuple[str, ...]:
        """Return the deduplicated asset keys required by the selected pipelines."""
        keys = [
            self.embedder_key,
            self.reranker_key,
            self.generator_key,
            self.picture_description_key,
            self.docling_artifacts_key,
            self.effective_chunk_tokenizer_key(),
        ]
        return tuple(dict.fromkeys(keys))


def default_pipeline_models() -> PipelineModels:
    """Return the default pipeline model selection."""
    return PipelineModels()


@dataclass(slots=True)
class ModelCatalog:
    """Registry of available local models and artifacts used by the application."""

    models_root: str = "models"

    hf_model_entries: tuple[ModelEntry, ...] = field(
        default_factory=lambda: (
            ModelEntry(
                key="qwen3_embedding_0_6b",
                repo_id="Qwen/Qwen3-Embedding-0.6B",
                relative_dir="embedders/qwen3-embedding-0.6b",
                role="embedder",
                selectable=True,
                label="Qwen3 Embedding 0.6B",
                description="Default dense embedding model for local retrieval.",
                size_hint="0.6B",
            ),
            ModelEntry(
                key="qwen3_reranker_0_6b",
                repo_id="Qwen/Qwen3-Reranker-0.6B",
                relative_dir="rerankers/qwen3-reranker-0.6b",
                role="reranker",
                label="Qwen3 Reranker 0.6B",
                description="Default reranker used internally by the retrieval pipeline.",
                size_hint="0.6B",
            ),
            ModelEntry(
                key="smolvlm_256m_instruct",
                repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
                relative_dir="vlm/smolvlm-256m-instruct",
                role="picture_description",
                label="SmolVLM 256M Instruct",
                description="Picture description model used internally by Docling when enabled.",
                size_hint="256M",
            ),
            ModelEntry(
                key="qwen3_4b",
                repo_id="Qwen/Qwen3-4B",
                relative_dir="generators/qwen3-4b",
                role="generator",
                selectable=True,
                label="Qwen3 4B",
                description="Official post-trained Qwen3 4B generator for general local use.",
                size_hint="4B",
                vram_hint="Up to 12 GB VRAM",
            ),
            ModelEntry(
                key="qwen3_4b_instruct_2507",
                repo_id="Qwen/Qwen3-4B-Instruct-2507",
                relative_dir="generators/qwen3-4b-instruct-2507",
                role="generator",
                selectable=True,
                label="Qwen3 4B Instruct 2507",
                description="Instruction-tuned generator optimized for grounded answers.",
                size_hint="4B",
                vram_hint="Up to 12 GB VRAM",
            ),
            ModelEntry(
                key="qwen3_1_7b",
                repo_id="Qwen/Qwen3-1.7B",
                relative_dir="generators/qwen3-1.7b",
                role="generator",
                selectable=True,
                label="Qwen3 1.7B",
                description="Official post-trained Qwen3 1.7B generator for mid-tier GPUs.",
                size_hint="1.7B",
                vram_hint="Up to 7 GB VRAM",
            ),
            ModelEntry(
                key="qwen3_0_6b",
                repo_id="Qwen/Qwen3-0.6B",
                relative_dir="generators/qwen3-0.6b",
                role="generator",
                selectable=True,
                label="Qwen3 0.6B",
                description="Official post-trained Qwen3 0.6B generator for constrained GPUs.",
                size_hint="0.6B",
                vram_hint="Up to 4 GB VRAM",
            ),
        )
    )

    artifact_entries: tuple[ArtifactEntry, ...] = field(
        default_factory=lambda: (
            ArtifactEntry(
                key="docling_artifacts",
                relative_dir="docling/artifacts",
                label="Docling artifacts",
                description="Offline parsing artifacts used by the local Docling pipeline.",
            ),
        )
    )

    def hf_models(self) -> tuple[ModelEntry, ...]:
        """Return all Hugging Face-backed model entries."""
        return self.hf_model_entries

    def artifacts(self) -> tuple[ArtifactEntry, ...]:
        """Return all non-Hugging Face artifact entries."""
        return self.artifact_entries

    def all(self) -> tuple[ModelEntry | ArtifactEntry, ...]:
        """Return all configured models and artifact bundles."""
        return self.hf_models() + self.artifacts()

    def downloadable_keys(self) -> tuple[str, ...]:
        """Return all keys supported by the downloader."""
        return tuple(entry.key for entry in self.all())

    def selectable_models(self, role: SelectableModelRole) -> tuple[ModelEntry, ...]:
        """Return setup-selectable models for one pipeline role."""
        return tuple(
            entry
            for entry in self.hf_models()
            if entry.role == role and entry.selectable
        )

    def generator_choices(self) -> list[dict[str, str | None]]:
        """Return setup metadata for selectable generator models."""
        return [entry.to_setup_option() for entry in self.selectable_models("generator")]

    def embedding_choices(self) -> list[dict[str, str | None]]:
        """Return setup metadata for selectable embedding models."""
        return [entry.to_setup_option() for entry in self.selectable_models("embedder")]

    def get(self, key: str) -> ModelEntry | ArtifactEntry:
        """Return a configured model or artifact entry by key."""
        for entry in self.all():
            if entry.key == key:
                return entry
        raise KeyError(f"Unknown model key: {key}")

    def get_hf_model(self, key: str) -> ModelEntry:
        """Return a Hugging Face model entry by key."""
        entry = self.get(key)
        if not isinstance(entry, ModelEntry):
            raise KeyError(f"Key '{key}' is not a Hugging Face model entry")
        return entry

    def get_artifact(self, key: str) -> ArtifactEntry:
        """Return a non-Hugging Face artifact entry by key."""
        entry = self.get(key)
        if not isinstance(entry, ArtifactEntry):
            raise KeyError(f"Key '{key}' is not an artifact entry")
        return entry

    def resolve_hf_path(self, key: str, project_root: str | Path) -> Path:
        """Resolve the local directory for a Hugging Face model key."""
        return self.get_hf_model(key).resolve_dir(project_root, self.models_root)

    def resolve_artifact_path(self, key: str, project_root: str | Path) -> Path:
        """Resolve the local directory for a non-Hugging Face artifact key."""
        return self.get_artifact(key).resolve_dir(project_root, self.models_root)

    def local_paths(
        self,
        project_root: str | Path,
        keys: tuple[str, ...] | None = None,
    ) -> dict[str, str]:
        """Return resolved local paths for selected configured assets."""
        selected_keys = set(keys or self.downloadable_keys())
        root = Path(project_root).resolve()
        return {
            entry.key: str(entry.resolve_dir(root, self.models_root))
            for entry in self.all()
            if entry.key in selected_keys
        }
