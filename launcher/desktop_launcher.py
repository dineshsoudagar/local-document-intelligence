"""Desktop bootstrap launcher for the managed local document intelligence app."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Thread

import uvicorn
import webview

from src.app.paths import APP_ROOT_ENV_VAR, CODE_ROOT_ENV_VAR, AppPaths
from src.app.runtime_state import load_managed_app_config


HOST = "127.0.0.1"
PORT = 8000
HEALTH_URL = f"http://{HOST}:{PORT}/healthz"
APP_URL = f"http://{HOST}:{PORT}/"


class BackendProcess:
    """Start and stop the local FastAPI backend for the desktop shell."""

    def __init__(self, paths: AppPaths) -> None:
        self._paths = paths
        self._process: subprocess.Popen[str] | None = None
        self._log_stream = None
        self._server: uvicorn.Server | None = None
        self._server_thread: Thread | None = None

    def start(self) -> None:
        """Start the backend process if it is not already running."""
        if self._process is not None and self._process.poll() is None:
            return
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        runtime_config = load_managed_app_config(self._paths.runtime_config_path)
        managed_python = self._paths.managed_venv_dir / "Scripts" / "python.exe"
        if runtime_config.install_state == "ready" and managed_python.is_file():
            self._start_managed_subprocess(managed_python)
            return

        self._start_embedded_server()

    def _start_embedded_server(self) -> None:
        """Run the bootstrap backend in-process for first-run setup."""
        os.environ[APP_ROOT_ENV_VAR] = str(self._paths.app_root)
        os.environ[CODE_ROOT_ENV_VAR] = str(self._paths.code_root)
        if str(self._paths.code_root) not in sys.path:
            sys.path.insert(0, str(self._paths.code_root))

        from src.api.main import app

        config = uvicorn.Config(
            app,
            host=HOST,
            port=PORT,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._server_thread = Thread(
            target=self._server.run,
            daemon=True,
            name="ldi-backend-server",
        )
        self._server_thread.start()

    def _start_managed_subprocess(self, backend_python: Path) -> None:
        """Start the managed backend process from the provisioned venv."""
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._paths.launcher_log_path
        env = os.environ.copy()
        env[APP_ROOT_ENV_VAR] = str(self._paths.app_root)
        env[CODE_ROOT_ENV_VAR] = str(self._paths.code_root)
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(self._paths.code_root)
            if not python_path
            else f"{self._paths.code_root}{os.pathsep}{python_path}"
        )

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting backend with {backend_python}\n"
            )

        self._log_stream = log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                str(backend_python),
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                HOST,
                "--port",
                str(PORT),
            ],
            cwd=str(self._paths.code_root),
            env=env,
            stdout=self._log_stream,
            stderr=self._log_stream,
            text=True,
        )

    def wait_until_ready(self, *, timeout_seconds: int = 90) -> None:
        """Block until the backend health endpoint is ready."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self._server_thread is not None and not self._server_thread.is_alive():
                raise RuntimeError("Embedded backend thread exited early.")
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Backend exited early with code {self._process.returncode}."
                )

            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass

            time.sleep(1)

        raise TimeoutError("Backend did not become ready in time.")

    def stop(self) -> None:
        """Stop the backend process if it is running."""
        if self._server is not None:
            self._server.should_exit = True
            if self._server_thread is not None:
                self._server_thread.join(timeout=20)
            self._server = None
            self._server_thread = None

        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)

        self._process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None


def _resolve_code_root() -> Path:
    """Return the launcher payload root."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Start the backend and host the UI in a desktop shell."""
    os.environ.setdefault(CODE_ROOT_ENV_VAR, str(_resolve_code_root()))
    paths = AppPaths.from_default_locations()
    backend = BackendProcess(paths)
    backend.start()
    backend.wait_until_ready()
    atexit.register(backend.stop)

    window = webview.create_window(
        title="Local Document Intelligence",
        url=APP_URL,
        width=1480,
        height=980,
        min_size=(1120, 760),
    )
    try:
        webview.start(debug=False)
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
