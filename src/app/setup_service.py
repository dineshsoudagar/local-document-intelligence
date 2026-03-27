"""Managed runtime provisioning and setup status orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.app.paths import AppPaths
from src.app.runtime_controller import RuntimeController
from src.app.runtime_state import (
    ManagedAppConfig,
    SetupStatus,
    load_managed_app_config,
    load_setup_status,
    save_managed_app_config,
    save_setup_status,
    utc_now_iso,
)
from src.config.generator_config import GeneratorConfig
from src.config.model_catalog import ModelCatalog, default_pipeline_models


class SetupCancelledError(RuntimeError):
    """Raised when a requested setup run is cancelled."""


@dataclass(frozen=True, slots=True)
class TorchVariantSpec:
    """One selectable PyTorch runtime variant."""

    key: str
    label: str
    description: str
    index_url: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-friendly metadata."""
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "index_url": self.index_url,
        }


TORCH_VERSION = "2.10.0"
TORCH_VISION_VERSION = "0.25.0"
TORCH_AUDIO_VERSION = "2.10.0"
TORCH_VARIANTS: dict[str, TorchVariantSpec] = {
    "cpu": TorchVariantSpec(
        key="cpu",
        label="CPU",
        description="CPU-only PyTorch build with the widest compatibility.",
        index_url="https://download.pytorch.org/whl/cpu",
    ),
    "cu128": TorchVariantSpec(
        key="cu128",
        label="NVIDIA CUDA 12.8",
        description="CUDA-enabled PyTorch build for NVIDIA GPUs.",
        index_url="https://download.pytorch.org/whl/cu128",
    ),
}


class SetupService:
    """Manage first-run setup, background provisioning, and persisted status."""

    def __init__(self, paths: AppPaths, runtime_controller: RuntimeController) -> None:
        self._paths = paths
        self._runtime_controller = runtime_controller
        self._catalog = ModelCatalog(models_root="models")
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._log_path = self._paths.logs_dir / "setup.log"

        self._paths.ensure_exists()
        status = load_setup_status(self._paths.setup_status_path)
        if status.install_state == "installing" and not status.is_busy:
            status = status.with_updates(
                install_state="failed",
                current_step="interrupted",
                progress_message="Setup was interrupted. Retry to continue.",
                last_error="Setup was interrupted before completion.",
                cancel_requested=False,
                is_busy=False,
            )
            save_setup_status(self._paths.setup_status_path, status)

    def get_status(self) -> SetupStatus:
        """Return the current persisted setup status."""
        status = load_setup_status(self._paths.setup_status_path)
        config = load_managed_app_config(self._paths.runtime_config_path)
        if self._runtime_controller.last_error and config.install_state == "ready":
            status = status.with_updates(
                install_state="failed",
                current_step="runtime_init_failed",
                progress_message="Runtime initialization failed.",
                last_error=self._runtime_controller.last_error,
                is_busy=False,
            )
            save_setup_status(self._paths.setup_status_path, status)
            return status

        if self._runtime_controller.is_ready() or config.install_state == "ready":
            if status.install_state != "ready" or status.is_busy:
                status = status.with_updates(
                    install_state="ready",
                    current_step="ready",
                    progress_message="Runtime is ready.",
                    last_error=None,
                    cancel_requested=False,
                    is_busy=False,
                    completed_at=status.completed_at or utc_now_iso(),
                )
                save_setup_status(self._paths.setup_status_path, status)
            return status
        return status

    def get_options(self) -> dict[str, Any]:
        """Return the setup choices needed by the frontend wizard."""
        compute = self._detect_compute()
        presets = []
        for preset in GeneratorConfig.available_load_presets():
            if preset["key"] in {"bnb_8bit", "bnb_4bit"} and not compute["cuda_available"]:
                continue
            if preset["key"] == "cpu_safe":
                continue
            presets.append(preset)

        return {
            "generator_models": self._catalog.generator_choices(),
            "embedding_models": self._catalog.embedding_choices(),
            "generator_load_presets": presets,
            "compute": compute,
            "torch_variants": [
                TORCH_VARIANTS[key].to_dict()
                for key in compute["allowed_torch_variants"]
            ],
        }

    def start_install(
        self,
        *,
        generator_key: str,
        embedding_key: str,
        generator_load_preset: str,
        torch_variant: str,
    ) -> SetupStatus:
        """Start a new managed runtime provisioning run."""
        self._validate_selection(
            generator_key=generator_key,
            embedding_key=embedding_key,
            generator_load_preset=generator_load_preset,
            torch_variant=torch_variant,
        )
        config = ManagedAppConfig(
            install_state="installing",
            selected_generator_key=generator_key,
            selected_embedding_key=embedding_key,
            selected_generator_load_preset=generator_load_preset,
            selected_torch_variant=torch_variant,
        )
        return self._start_worker(config)

    def retry_install(self) -> SetupStatus:
        """Retry provisioning with the last persisted selections."""
        config = load_managed_app_config(self._paths.runtime_config_path)
        if (
            not config.selected_generator_key
            or not config.selected_embedding_key
            or not config.selected_generator_load_preset
            or not config.selected_torch_variant
        ):
            raise ValueError("No previous setup selection exists to retry.")

        retry_config = config.model_copy(update={"install_state": "installing"})
        self._validate_selection(
            generator_key=retry_config.selected_generator_key,
            embedding_key=retry_config.selected_embedding_key,
            generator_load_preset=retry_config.selected_generator_load_preset,
            torch_variant=retry_config.selected_torch_variant,
        )
        return self._start_worker(retry_config)

    def cancel_install(self) -> SetupStatus:
        """Request cancellation for the active setup run."""
        with self._lock:
            status = load_setup_status(self._paths.setup_status_path)
            status = status.with_updates(
                cancel_requested=True,
                progress_message="Cancelling setup...",
            )
            save_setup_status(self._paths.setup_status_path, status)
            process = self._active_process

        if process is not None and process.poll() is None:
            process.terminate()

        return status

    def _start_worker(self, config: ManagedAppConfig) -> SetupStatus:
        """Persist the initial config/status and start the background worker."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("Setup is already running.")

            self._runtime_controller.close()
            persisted_config = save_managed_app_config(self._paths.runtime_config_path, config)
            status = SetupStatus(
                install_state="installing",
                current_step="queued",
                progress_message="Preparing managed runtime...",
                last_error=None,
                cancel_requested=False,
                is_busy=True,
                selected_generator_key=persisted_config.selected_generator_key,
                selected_embedding_key=persisted_config.selected_embedding_key,
                selected_generator_load_preset=persisted_config.selected_generator_load_preset,
                selected_torch_variant=persisted_config.selected_torch_variant,
                started_at=utc_now_iso(),
                completed_at=None,
            )
            save_setup_status(self._paths.setup_status_path, status)

            self._worker = threading.Thread(
                target=self._run_install,
                args=(persisted_config,),
                daemon=True,
                name="ldi-setup-worker",
            )
            self._worker.start()

        return status

    def _run_install(self, config: ManagedAppConfig) -> None:
        """Provision the managed runtime in a background worker thread."""
        try:
            self._write_status(
                current_step="prepare_directories",
                progress_message="Creating managed runtime directories...",
            )
            self._paths.ensure_exists()
            self._check_cancel_requested()

            bootstrap_python = self._resolve_bootstrap_python()
            managed_python = self._paths.managed_venv_dir / "Scripts" / "python.exe"

            if not managed_python.is_file():
                self._write_status(
                    current_step="create_venv",
                    progress_message="Creating managed virtual environment...",
                )
                self._run_process(
                    [str(bootstrap_python), "-m", "venv", str(self._paths.managed_venv_dir)],
                    cwd=self._paths.code_root,
                )
            self._check_cancel_requested()

            self._write_status(
                current_step="upgrade_pip",
                progress_message="Upgrading pip in the managed runtime...",
            )
            self._run_process(
                [str(managed_python), "-m", "pip", "install", "--upgrade", "pip"],
                cwd=self._paths.code_root,
            )
            self._check_cancel_requested()

            self._write_status(
                current_step="install_base_requirements",
                progress_message="Installing bundled base Python packages...",
            )
            self._install_base_requirements(managed_python)
            self._check_cancel_requested()

            self._write_status(
                current_step="install_torch",
                progress_message="Installing PyTorch runtime...",
            )
            self._install_torch(managed_python, config.selected_torch_variant or "cpu")
            self._check_cancel_requested()

            if config.selected_generator_load_preset in {"bnb_8bit", "bnb_4bit"}:
                self._write_status(
                    current_step="install_bitsandbytes",
                    progress_message="Installing bitsandbytes for low-memory loading...",
                )
                self._run_process(
                    [str(managed_python), "-m", "pip", "install", "bitsandbytes"],
                    cwd=self._paths.code_root,
                )
                self._check_cancel_requested()

            self._write_status(
                current_step="download_models",
                progress_message="Downloading selected models and offline artifacts...",
            )
            self._download_selected_models(managed_python, config)
            self._check_cancel_requested()

            save_managed_app_config(
                self._paths.runtime_config_path,
                config.model_copy(update={"install_state": "ready"}),
            )
            self._runtime_controller.reload()
            self._write_status(
                install_state="ready",
                current_step="ready",
                progress_message="Runtime is ready.",
                last_error=None,
                cancel_requested=False,
                is_busy=False,
                completed_at=utc_now_iso(),
            )
        except SetupCancelledError:
            save_managed_app_config(
                self._paths.runtime_config_path,
                config.model_copy(update={"install_state": "not_ready"}),
            )
            self._write_status(
                install_state="not_ready",
                current_step="cancelled",
                progress_message="Setup cancelled.",
                is_busy=False,
                cancel_requested=False,
                completed_at=utc_now_iso(),
            )
        except Exception as exc:
            save_managed_app_config(
                self._paths.runtime_config_path,
                config.model_copy(update={"install_state": "failed"}),
            )
            self._write_status(
                install_state="failed",
                current_step="failed",
                progress_message="Setup failed.",
                last_error=str(exc),
                cancel_requested=False,
                is_busy=False,
                completed_at=utc_now_iso(),
            )
        finally:
            with self._lock:
                self._active_process = None

    def _install_base_requirements(self, managed_python: Path) -> None:
        """Install the non-torch runtime requirements into the managed venv."""
        wheelhouse = self._paths.bundled_wheelhouse_dir
        args = [str(managed_python), "-m", "pip", "install", "-r", str(self._paths.requirements_path)]
        if wheelhouse.is_dir() and any(wheelhouse.iterdir()):
            args = [
                str(managed_python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                "-r",
                str(self._paths.requirements_path),
            ]
        self._run_process(args, cwd=self._paths.code_root)

    def _install_torch(self, managed_python: Path, torch_variant: str) -> None:
        """Install the selected PyTorch variant into the managed runtime."""
        spec = TORCH_VARIANTS[torch_variant]
        args = [
            str(managed_python),
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCH_VISION_VERSION}",
            f"torchaudio=={TORCH_AUDIO_VERSION}",
            "--index-url",
            spec.index_url,
        ]
        self._run_process(args, cwd=self._paths.code_root)

    def _download_selected_models(self, managed_python: Path, config: ManagedAppConfig) -> None:
        """Download the selected public models plus fixed internal defaults."""
        defaults = default_pipeline_models()
        only_keys = [
            config.selected_generator_key,
            config.selected_embedding_key,
            defaults.reranker_key,
            defaults.docling_artifacts_key,
        ]
        deduped_keys = [key for key in dict.fromkeys(only_keys) if key]
        args = [
            str(managed_python),
            str(self._paths.download_models_script_path),
            "--project-root",
            str(self._paths.app_root),
            "--only",
            *deduped_keys,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._paths.code_root)
        self._run_process(args, cwd=self._paths.code_root, env=env)

    def _run_process(
        self,
        args: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        """Run one subprocess while supporting cancellation and setup logging."""
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{utc_now_iso()}] RUN {' '.join(args)}\n")
            log_file.flush()
            process = subprocess.Popen(
                args,
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=log_file,
                text=True,
            )
            with self._lock:
                self._active_process = process

            while True:
                if process.poll() is not None:
                    break
                if self._cancel_requested():
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    raise SetupCancelledError("Setup was cancelled by the user.")
                time.sleep(0.5)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {' '.join(args)}"
                )

    def _resolve_bootstrap_python(self) -> Path:
        """Return the Python executable used to create the managed venv."""
        bundled_python = self._paths.bundled_python_dir / "python.exe"
        if bundled_python.is_file():
            return bundled_python
        embedded_python = self._paths.embedded_python_dir / "python.exe"
        if embedded_python.is_file():
            return embedded_python
        return Path(sys.executable).resolve()

    def _write_status(self, **updates: object) -> SetupStatus:
        """Persist one status update."""
        status = load_setup_status(self._paths.setup_status_path)
        updated = status.with_updates(**updates)
        return save_setup_status(self._paths.setup_status_path, updated)

    def _cancel_requested(self) -> bool:
        """Return whether the persisted status requested cancellation."""
        status = load_setup_status(self._paths.setup_status_path)
        return bool(status.cancel_requested)

    def _check_cancel_requested(self) -> None:
        """Raise if cancellation was requested before starting a new step."""
        if self._cancel_requested():
            raise SetupCancelledError("Setup was cancelled by the user.")

    def _detect_compute(self) -> dict[str, Any]:
        """Return lightweight compute recommendations without importing torch."""
        cuda_available = False
        gpu_name: str | None = None
        gpu_memory_gb: int | None = None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines:
                    cuda_available = True
                    first_line = lines[0]
                    parts = [part.strip() for part in first_line.split(",")]
                    gpu_name = parts[0] if parts else None
                    if len(parts) > 1:
                        try:
                            gpu_memory_mb = int(parts[1])
                            gpu_memory_gb = max(1, round(gpu_memory_mb / 1024))
                        except ValueError:
                            gpu_memory_gb = None
        except Exception:
            cuda_available = False
            gpu_name = None
            gpu_memory_gb = None

        recommended_torch_variant = "cu128" if cuda_available else "cpu"
        recommended_generator_load_preset = "standard"
        if cuda_available:
            if gpu_memory_gb is not None and gpu_memory_gb <= 6:
                recommended_generator_load_preset = "bnb_4bit"
            elif gpu_memory_gb is not None and gpu_memory_gb <= 8:
                recommended_generator_load_preset = "bnb_8bit"
            else:
                recommended_generator_load_preset = "standard"
        return {
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "recommended_torch_variant": recommended_torch_variant,
            "recommended_generator_load_preset": recommended_generator_load_preset,
            "allowed_torch_variants": ["cpu", "cu128"] if cuda_available else ["cpu"],
        }

    def _validate_selection(
        self,
        *,
        generator_key: str,
        embedding_key: str,
        generator_load_preset: str,
        torch_variant: str,
    ) -> None:
        """Validate a setup request against the curated catalog and compute rules."""
        generator_keys = {
            option["key"]
            for option in self._catalog.generator_choices()
        }
        embedding_keys = {
            option["key"]
            for option in self._catalog.embedding_choices()
        }
        compute = self._detect_compute()
        preset_keys = {
            preset["key"]
            for preset in self.get_options()["generator_load_presets"]
        }

        if generator_key not in generator_keys:
            raise ValueError(f"Unsupported generator model key: {generator_key}")
        if embedding_key not in embedding_keys:
            raise ValueError(f"Unsupported embedding model key: {embedding_key}")
        if generator_load_preset not in preset_keys:
            raise ValueError(
                f"Unsupported generator load preset: {generator_load_preset}"
            )
        if torch_variant not in compute["allowed_torch_variants"]:
            raise ValueError(f"Unsupported torch variant for this machine: {torch_variant}")
        if generator_load_preset in {"bnb_8bit", "bnb_4bit"} and torch_variant == "cpu":
            raise ValueError("Bitsandbytes presets require an NVIDIA CUDA torch variant.")
