"""Lazy runtime initialization for the document intelligence service stack."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from src.app.document_registry import DocumentRegistry
from src.app.paths import AppPaths
from src.app.runtime_state import ManagedAppConfig, load_managed_app_config
from src.config.generator_config import GeneratorConfig
from src.config.index_config import IndexConfig
from src.config.model_catalog import ModelCatalog, default_pipeline_models
from src.config.parser_config import ParserConfig


@dataclass(slots=True)
class RuntimeServices:
    """Live service instances used by the API routes."""

    config: ManagedAppConfig
    document_registry: DocumentRegistry
    index_service: object
    answer_service: object


class RuntimeController:
    """Load and manage the heavyweight retrieval/generation runtime lazily."""

    def __init__(self, paths: AppPaths) -> None:
        self._paths = paths
        self._services: RuntimeServices | None = None
        self._config = load_managed_app_config(paths.runtime_config_path)
        self._last_error: str | None = None

    @property
    def config(self) -> ManagedAppConfig:
        """Return the last persisted runtime config."""
        return self._config

    @property
    def services(self) -> RuntimeServices | None:
        """Return initialized runtime services when available."""
        return self._services

    @property
    def last_error(self) -> str | None:
        """Return the last runtime initialization error."""
        return self._last_error

    def is_ready(self) -> bool:
        """Return whether the heavyweight runtime is initialized."""
        return self._services is not None

    def refresh_config(self) -> ManagedAppConfig:
        """Reload the persisted config from disk."""
        self._config = load_managed_app_config(self._paths.runtime_config_path)
        return self._config

    def initialize_if_ready(self) -> bool:
        """Initialize runtime services if the persisted config is ready."""
        self.refresh_config()
        if self._config.install_state != "ready":
            self.close()
            return False

        self._services = self._build_services(self._config)
        self._last_error = None
        return True

    def reload(self) -> bool:
        """Rebuild services from the current persisted config."""
        self.close()
        try:
            return self.initialize_if_ready()
        except Exception as exc:
            self._last_error = str(exc)
            self._services = None
            return False

    def close(self) -> None:
        """Release initialized services."""
        if self._services is None:
            return

        index_service = self._services.index_service
        close_method = getattr(index_service, "close", None)
        if callable(close_method):
            close_method()

        self._services = None

    def _build_services(self, config: ManagedAppConfig) -> RuntimeServices:
        """Create the heavyweight application services from one runtime config."""
        from docling.datamodel.base_models import InputFormat

        from src.generation.answer_service import GroundedAnswerService
        from src.indexing.index_service import IndexService
        from src.retrieval.qwen_models import LocalQwenGenerator

        self._paths.ensure_exists()

        registry = DocumentRegistry(self._paths.documents_db_path)
        registry.initialize()

        pipeline_models = default_pipeline_models()
        if config.selected_generator_key:
            pipeline_models = replace(
                pipeline_models,
                generator_key=config.selected_generator_key,
            )
        if config.selected_embedding_key:
            pipeline_models = replace(
                pipeline_models,
                embedder_key=config.selected_embedding_key,
            )

        model_catalog = ModelCatalog(models_root="models")
        generator_config = GeneratorConfig(
            project_root=self._paths.app_root,
            model_catalog=model_catalog,
            pipeline_models=pipeline_models,
        )
        preset_key = config.selected_generator_load_preset or "standard"
        generator_config = generator_config.with_load_preset(preset_key)

        index_config = IndexConfig(
            project_root=self._paths.app_root,
            model_catalog=model_catalog,
            pipeline_models=pipeline_models,
            qdrant_path_override=self._paths.qdrant_dir,
        )
        parser_config = ParserConfig(
            project_root=self._paths.app_root,
            model_catalog=model_catalog,
            pipeline_models=pipeline_models,
            allowed_formats=[InputFormat.PDF],
            enable_picture_description=False,
            include_picture_chunks=True,
        )
        index_service = IndexService(
            index_config=index_config,
            parser_config=parser_config,
            generator_config=generator_config,
        )
        try:
            index_service.warm_up_parser()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Parser warmup failed during runtime startup: %s",
                exc,
            )

        generator = LocalQwenGenerator(generator_config.generator_model_path, config=generator_config)
        answer_service = GroundedAnswerService(
            index=index_service.index,
            config=generator_config,
            generator=generator,
        )

        return RuntimeServices(
            config=config,
            document_registry=registry,
            index_service=index_service,
            answer_service=answer_service,
        )
