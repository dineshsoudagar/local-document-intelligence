"""Managed runtime provisioning and setup status orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.app.paths import (
    BACKEND_RUNTIME_MODE_ENV_VAR,
    CODE_ROOT_ENV_VAR,
    AppPaths,
)
from src.app.python_runtime import (
    ensure_embedded_python_runtime,
    hidden_windows_subprocess_kwargs,
    python_executable_is_usable,
    sanitized_subprocess_env,
)
from src.app.runtime_controller_like import RuntimeControllerLike
from src.app.runtime_state import (
    ManagedAppConfig,
    SetupProgressItem,
    SetupStatus,
    load_managed_app_config,
    load_setup_status,
    save_managed_app_config,
    save_setup_status,
    utc_now_iso,
)
from src.config.generator_config import GeneratorConfig
from src.config.model_catalog import ArtifactEntry, ModelCatalog, ModelEntry, default_pipeline_models


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

PROGRESS_EVENT_PREFIX = "LDI_PROGRESS "
OVERALL_PACKAGE_WEIGHT = 55
OVERALL_MODEL_WEIGHT = 45


class SetupService:
    """Manage first-run setup, background provisioning, and persisted status."""

    def __init__(self, paths: AppPaths, runtime_controller: RuntimeControllerLike) -> None:
        self._paths = paths
        self._runtime_controller = runtime_controller
        self._backend_runtime_mode = os.getenv(BACKEND_RUNTIME_MODE_ENV_VAR, "unknown")
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
        if (
            self._runtime_controller.last_error
            and config.install_state == "ready"
            and not status.is_busy
        ):
            status = status.with_updates(
                install_state="failed",
                current_step="runtime_init_failed",
                progress_message="Runtime initialization failed.",
                last_error=self._runtime_controller.last_error,
                is_busy=False,
            )
            save_setup_status(self._paths.setup_status_path, status)
            return status

        if self._runtime_controller.is_ready() or (
            config.install_state == "ready" and not status.is_busy
        ):
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
        logging.getLogger(__name__).info(
            "Setup requested generator_key=%s embedding_key=%s "
            "generator_load_preset=%s torch_variant=%s",
            generator_key,
            embedding_key,
            generator_load_preset,
            torch_variant,
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
        logging.getLogger(__name__).info(
            "Setup retry requested generator_key=%s embedding_key=%s "
            "generator_load_preset=%s torch_variant=%s",
            retry_config.selected_generator_key,
            retry_config.selected_embedding_key,
            retry_config.selected_generator_load_preset,
            retry_config.selected_torch_variant,
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

    def _selected_download_entries(
        self,
        config: ManagedAppConfig,
    ) -> list[ModelEntry | ArtifactEntry]:
        """Return the deduplicated assets the installer will materialize."""
        defaults = default_pipeline_models()
        selected_keys = [
            config.selected_generator_key,
            config.selected_embedding_key,
            defaults.reranker_key,
            defaults.docling_artifacts_key,
        ]
        deduped_keys = [key for key in dict.fromkeys(selected_keys) if key]
        return [self._catalog.get(key) for key in deduped_keys]

    def _build_model_progress_items(
        self,
        config: ManagedAppConfig,
    ) -> list[SetupProgressItem]:
        """Build pending progress rows for each required model or artifact."""
        items: list[SetupProgressItem] = []
        for entry in self._selected_download_entries(config):
            label = entry.label if getattr(entry, "label", None) else entry.key
            detail = getattr(entry, "repo_id", None)
            items.append(
                SetupProgressItem(
                    key=entry.key,
                    label=label,
                    status="pending",
                    progress=0,
                    detail=detail,
                )
            )
        return items

    def _normalize_status_progress(self, status: SetupStatus) -> SetupStatus:
        """Derive consistent overall and nested progress values for the UI."""
        package_progress = max(0, min(100, status.package_progress))
        items: list[SetupProgressItem] = []
        for item in status.model_progress_items:
            progress = max(0, min(100, item.progress))
            if item.status in {"complete", "skipped"}:
                progress = 100
            items.append(item.model_copy(update={"progress": progress}))

        if status.install_state == "ready":
            items = [
                item
                if item.status in {"complete", "skipped"}
                else item.model_copy(update={"status": "complete", "progress": 100})
                for item in items
            ]
            return status.model_copy(
                update={
                    "overall_progress": 100,
                    "package_progress": 100,
                    "package_message": status.package_message
                    or "Managed runtime packages installed.",
                    "model_progress_items": items,
                }
            )

        model_progress = 0
        if items:
            model_progress = round(
                sum(item.progress for item in items) / len(items)
            )

        overall_progress = package_progress
        if items:
            overall_progress = round(
                (package_progress * OVERALL_PACKAGE_WEIGHT / 100)
                + (model_progress * OVERALL_MODEL_WEIGHT / 100)
            )

        return status.model_copy(
            update={
                "overall_progress": max(0, min(100, overall_progress)),
                "package_progress": package_progress,
                "model_progress_items": items,
            }
        )

    def _set_model_progress_item(
        self,
        key: str,
        *,
        status: str,
        progress: int,
        detail: str | None = None,
        progress_message: str | None = None,
    ) -> SetupStatus:
        """Update one model/artifact progress row and persist the result."""
        current = load_setup_status(self._paths.setup_status_path)
        updated_items: list[SetupProgressItem] = []
        for item in current.model_progress_items:
            if item.key == key:
                updated_items.append(
                    item.model_copy(
                        update={
                            "status": status,
                            "progress": progress,
                            "detail": detail if detail is not None else item.detail,
                        }
                    )
                )
            else:
                updated_items.append(item)

        payload: dict[str, object] = {"model_progress_items": updated_items}
        if progress_message is not None:
            payload["progress_message"] = progress_message
        return self._write_status(**payload)

    def _mark_running_model_items_failed(self, detail: str) -> None:
        """Mark any in-flight model rows as failed after a downloader error."""
        status = load_setup_status(self._paths.setup_status_path)
        updated_items = [
            item.model_copy(update={"status": "failed", "detail": detail})
            if item.status == "running"
            else item
            for item in status.model_progress_items
        ]
        self._write_status(model_progress_items=updated_items)

    def _handle_download_progress_line(self, line: str) -> None:
        """Apply structured downloader progress events to persisted setup status."""
        if not line.startswith(PROGRESS_EVENT_PREFIX):
            return

        try:
            payload = json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
        except json.JSONDecodeError:
            return

        event = payload.get("event")
        key = payload.get("key")
        if not isinstance(key, str):
            return

        label = payload.get("label") if isinstance(payload.get("label"), str) else key
        if event == "asset_skip":
            self._set_model_progress_item(
                key,
                status="skipped",
                progress=100,
                detail="Already available locally.",
                progress_message=f"{label} is already available locally.",
            )
            return

        if event == "asset_start":
            detail = payload.get("detail") if isinstance(payload.get("detail"), str) else None
            self._set_model_progress_item(
                key,
                status="running",
                progress=35,
                detail=detail,
                progress_message=f"Downloading {label}...",
            )
            return

        if event == "asset_complete":
            target_dir = payload.get("target_dir")
            detail = target_dir if isinstance(target_dir, str) else "Saved locally."
            self._set_model_progress_item(
                key,
                status="complete",
                progress=100,
                detail=detail,
                progress_message=f"Finished downloading {label}.",
            )

    def _start_worker(self, config: ManagedAppConfig) -> SetupStatus:
        """Persist the initial config/status and start the background worker."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("Setup is already running.")

            self._runtime_controller.close()
            self._runtime_controller.clear_error()
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
                overall_progress=0,
                package_progress=0,
                package_message="Queued",
                model_progress_items=self._build_model_progress_items(persisted_config),
                started_at=utc_now_iso(),
                completed_at=None,
            )
            save_setup_status(
                self._paths.setup_status_path,
                self._normalize_status_progress(status),
            )

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
                package_progress=5,
                package_message="Preparing managed runtime directories...",
            )
            self._paths.ensure_exists()
            self._check_cancel_requested()

            bootstrap_python = self._resolve_bootstrap_python()
            managed_python = self._ensure_managed_python(bootstrap_python)
            self._check_cancel_requested()

            self._write_status(
                current_step="upgrade_pip",
                progress_message="Upgrading pip in the managed runtime...",
                package_progress=34,
                package_message="Upgrading pip in the managed environment...",
            )
            self._run_process(
                [str(managed_python), "-m", "pip", "install", "--upgrade", "pip"],
                cwd=self._paths.code_root,
            )
            self._check_cancel_requested()

            self._write_status(
                current_step="install_base_requirements",
                progress_message="Installing bundled base Python packages...",
                package_progress=62,
                package_message="Installing bundled Python dependencies...",
            )
            self._install_base_requirements(managed_python)
            self._check_cancel_requested()

            self._write_status(
                current_step="install_torch",
                progress_message="Installing PyTorch runtime...",
                package_progress=84,
                package_message="Installing the selected PyTorch runtime...",
            )
            self._install_torch(managed_python, config.selected_torch_variant or "cpu")
            self._check_cancel_requested()

            if config.selected_generator_load_preset in {"bnb_8bit", "bnb_4bit"}:
                self._write_status(
                    current_step="install_bitsandbytes",
                    progress_message="Installing bitsandbytes for low-memory loading...",
                    package_progress=92,
                    package_message="Installing optional bitsandbytes support...",
                )
                self._run_process(
                    [
                        str(managed_python),
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "bitsandbytes",
                    ],
                    cwd=self._paths.code_root,
                )
                self._check_cancel_requested()

            self._write_status(
                current_step="download_models",
                progress_message="Downloading selected models and offline artifacts...",
                package_progress=100,
                package_message="Managed runtime packages installed.",
            )
            self._download_selected_models(managed_python, config)
            self._check_cancel_requested()

            save_managed_app_config(
                self._paths.runtime_config_path,
                config.model_copy(update={"install_state": "ready"}),
            )
            if self._should_defer_runtime_reload_to_managed_handoff():
                self._runtime_controller.close()
                self._runtime_controller.clear_error()
            else:
                self._runtime_controller.reload()
            self._write_status(
                install_state="ready",
                current_step="ready",
                progress_message=(
                    "Setup complete. Switching to the managed runtime..."
                    if self._should_defer_runtime_reload_to_managed_handoff()
                    else "Runtime is ready."
                ),
                package_progress=100,
                package_message="Managed runtime packages installed.",
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
            self._mark_running_model_items_failed(str(exc))
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
        staged_script = self._stage_setup_script(self._paths.download_models_script_path)
        args = [
            str(managed_python),
            str(staged_script),
            "--project-root",
            str(self._paths.app_root),
            "--only",
            *deduped_keys,
        ]
        self._run_process(
            args,
            cwd=self._paths.app_root,
            env={CODE_ROOT_ENV_VAR: str(self._paths.code_root)},
            line_callback=self._handle_download_progress_line,
        )

    def _stage_setup_script(self, source_path: Path) -> Path:
        """Copy one packaged setup script to a neutral runtime directory before execution."""
        staged_dir = self._paths.runtime_dir / "setup-scripts"
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staged_dir / source_path.name
        shutil.copyfile(source_path, staged_path)
        return staged_path

    def _run_process(
        self,
        args: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        line_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Run one subprocess while supporting cancellation and setup logging."""
        process_env = sanitized_subprocess_env(env)

        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{utc_now_iso()}] RUN {' '.join(args)}\n")
            log_file.flush()
            stdout_target: int | Any = log_file
            stderr_target: int | Any = log_file
            if line_callback is not None:
                stdout_target = subprocess.PIPE
                stderr_target = subprocess.STDOUT
            process = subprocess.Popen(
                args,
                cwd=str(cwd),
                env=process_env,
                stdout=stdout_target,
                stderr=stderr_target,
                text=True,
                bufsize=1,
                **hidden_windows_subprocess_kwargs(),
            )
            with self._lock:
                self._active_process = process

            reader_thread: threading.Thread | None = None
            if line_callback is not None and process.stdout is not None:
                def stream_output() -> None:
                    for raw_line in process.stdout:
                        log_file.write(raw_line)
                        log_file.flush()
                        line = raw_line.rstrip()
                        if not line:
                            continue
                        try:
                            line_callback(line)
                        except Exception as exc:  # pragma: no cover - best effort logging
                            log_file.write(
                                f"[{utc_now_iso()}] Progress callback error: {exc}\n"
                            )
                            log_file.flush()

                reader_thread = threading.Thread(
                    target=stream_output,
                    daemon=True,
                    name="ldi-setup-log-stream",
                )
                reader_thread.start()

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

            if reader_thread is not None:
                reader_thread.join(timeout=10)

            if process.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {process.returncode}: {' '.join(args)}"
                )

    def _resolve_bootstrap_python(self) -> Path:
        """Return the Python executable used to create the managed venv."""
        embedded_python = self._paths.embedded_python_dir / "python.exe"
        bundled_python = self._paths.bundled_python_dir / "python.exe"
        if bundled_python.is_file():
            embedded_python = ensure_embedded_python_runtime(
                bundled_python_dir=self._paths.bundled_python_dir,
                embedded_python_dir=self._paths.embedded_python_dir,
            )
            usable, detail = python_executable_is_usable(embedded_python)
            if usable:
                return embedded_python

            detail_suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                "Persistent bootstrap Python runtime is unusable after being copied "
                f"from the packaged bundle{detail_suffix}"
            )

        if embedded_python.is_file():
            return embedded_python
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "Packaged bootstrap Python runtime not found. "
                f"Expected '{bundled_python}' or '{embedded_python}'. "
                "Rebuild the desktop bundle so it includes bundle/python instead of "
                "falling back to the launcher executable."
            )
        return Path(sys.executable).resolve()

    def _append_log_line(self, message: str) -> None:
        """Append one diagnostic line to the setup log."""
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{utc_now_iso()}] {message}\n")

    def _ensure_managed_python(self, bootstrap_python: Path) -> Path:
        """Return a usable managed venv Python, recreating the venv if needed."""
        managed_python = self._paths.managed_venv_dir / "Scripts" / "python.exe"
        usable, detail = python_executable_is_usable(managed_python)
        if usable:
            return managed_python

        if self._paths.managed_venv_dir.exists():
            detail_suffix = f" Reason: {detail}" if detail else ""
            self._append_log_line(
                f"Removing unusable managed virtual environment at "
                f"{self._paths.managed_venv_dir}.{detail_suffix}"
            )
            shutil.rmtree(self._paths.managed_venv_dir)

        self._write_status(
            current_step="create_venv",
            progress_message="Creating managed virtual environment...",
            package_progress=18,
            package_message="Creating the managed Python environment...",
        )
        self._run_process(
            [str(bootstrap_python), "-m", "venv", str(self._paths.managed_venv_dir)],
            cwd=self._paths.code_root,
        )

        recreated_python = self._paths.managed_venv_dir / "Scripts" / "python.exe"
        usable, detail = python_executable_is_usable(recreated_python)
        if usable:
            return recreated_python

        detail_suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"Managed Python runtime is unusable after virtual environment creation"
            f"{detail_suffix}"
        )

    def _write_status(self, **updates: object) -> SetupStatus:
        """Persist one status update."""
        status = load_setup_status(self._paths.setup_status_path)
        updated = status.with_updates(**updates)
        return save_setup_status(
            self._paths.setup_status_path,
            self._normalize_status_progress(updated),
        )

    def _should_defer_runtime_reload_to_managed_handoff(self) -> bool:
        """Return whether setup should restart into a fresh managed process."""
        # The managed subprocess installs packages into the same venv it is
        # currently running from. A local reload in that live process can keep
        # stale import-time capability checks from libraries like transformers,
        # so the handoff must happen in a fresh interpreter.
        if self._backend_runtime_mode == "managed_subprocess":
            return True

        return getattr(sys, "frozen", False) and self._backend_runtime_mode == "embedded"

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
                **hidden_windows_subprocess_kwargs(),
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
