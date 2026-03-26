from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Definition for a Hugging Face model stored under the project models directory."""

    key: str
    repo_id: str
    relative_dir: str
    revision: str = "main"

    def resolve_dir(self, project_root: str | Path, models_root: str) -> Path:
        """Return the resolved local directory for this model."""
        return Path(project_root).resolve() / models_root / self.relative_dir


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """Definition for non-Hugging Face assets stored under the project models directory."""

    key: str
    relative_dir: str

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
            ),
            ModelEntry(
                key="qwen3_reranker_0_6b",
                repo_id="Qwen/Qwen3-Reranker-0.6B",
                relative_dir="rerankers/qwen3-reranker-0.6b",
            ),
            ModelEntry(
                key="qwen3_4b_instruct_2507",
                repo_id="Qwen/Qwen3-4B-Instruct-2507",
                relative_dir="generators/qwen3-4b-instruct-2507",
            ),
            ModelEntry(
                key="smolvlm_256m_instruct",
                repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
                relative_dir="vlm/smolvlm-256m-instruct",
            ),
            ModelEntry(
                key="qwen3_4b",
                repo_id="Qwen/Qwen3-4B",
                relative_dir="generators/qwen3-4b",
            ),
        )
    )

    artifact_entries: tuple[ArtifactEntry, ...] = field(
        default_factory=lambda: (
            ArtifactEntry(
                key="docling_artifacts",
                relative_dir="docling/artifacts",
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