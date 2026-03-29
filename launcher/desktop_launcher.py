"""Desktop bootstrap launcher for the managed local document intelligence app."""

from __future__ import annotations

import atexit
import html
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Thread
from textwrap import dedent

import uvicorn
import webview

from src.app.paths import (
    APP_ROOT_ENV_VAR,
    BACKEND_RUNTIME_MODE_ENV_VAR,
    CODE_ROOT_ENV_VAR,
    LAUNCHER_LOG_PATH_ENV_VAR,
    AppPaths,
)
from src.app.python_runtime import (
    hidden_windows_subprocess_kwargs,
    python_executable_is_usable,
    sanitized_subprocess_env,
)
from src.app.runtime_state import load_managed_app_config


HOST = "127.0.0.1"
PORT = 8000
HEALTH_URL = f"http://{HOST}:{PORT}/healthz"
READY_URL = f"http://{HOST}:{PORT}/readyz"
APP_URL = f"http://{HOST}:{PORT}/"


def _render_startup_html(
    *,
    heading: str = "Starting your local workspace",
    message: str = "Preparing the local backend and runtime. This window will continue automatically.",
) -> str:
    """Return a small branded loading screen shown before the app is ready."""
    heading_text = html.escape(heading)
    message_text = html.escape(message)
    return dedent(
        f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Local Document Intelligence</title>
            <style>
              :root {{
                color-scheme: light;
                --bg: #eef5ff;
                --panel: rgba(255, 255, 255, 0.82);
                --text: #183153;
                --muted: #58739b;
                --accent: #2b7fff;
                --accent-soft: rgba(43, 127, 255, 0.16);
                --border: rgba(43, 127, 255, 0.14);
              }}
              * {{
                box-sizing: border-box;
              }}
              body {{
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                font-family: "Segoe UI", "Inter", sans-serif;
                color: var(--text);
                background:
                  radial-gradient(circle at top left, rgba(43, 127, 255, 0.16), transparent 32%),
                  radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.14), transparent 28%),
                  linear-gradient(180deg, #f7fbff 0%, var(--bg) 100%);
              }}
              .shell {{
                width: min(560px, calc(100vw - 48px));
                padding: 40px 36px;
                border-radius: 28px;
                background: var(--panel);
                border: 1px solid var(--border);
                box-shadow: 0 24px 64px rgba(24, 49, 83, 0.12);
                backdrop-filter: blur(14px);
              }}
              .eyebrow {{
                margin: 0 0 10px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                color: var(--accent);
              }}
              h1 {{
                margin: 0;
                font-size: clamp(28px, 4vw, 38px);
                line-height: 1.08;
              }}
              p {{
                margin: 16px 0 0;
                font-size: 15px;
                line-height: 1.6;
                color: var(--muted);
              }}
              .status {{
                margin-top: 26px;
                display: flex;
                align-items: center;
                gap: 14px;
                padding: 15px 18px;
                border-radius: 18px;
                background: var(--accent-soft);
              }}
              .spinner {{
                width: 18px;
                height: 18px;
                border-radius: 999px;
                border: 3px solid rgba(43, 127, 255, 0.2);
                border-top-color: var(--accent);
                animation: spin 0.9s linear infinite;
                flex: 0 0 auto;
              }}
              .status span {{
                font-size: 14px;
                font-weight: 600;
                color: var(--text);
              }}
              @keyframes spin {{
                to {{
                  transform: rotate(360deg);
                }}
              }}
            </style>
          </head>
          <body>
            <main class="shell">
              <p class="eyebrow">Local Document Intelligence</p>
              <h1>{heading_text}</h1>
              <p>{message_text}</p>
              <div class="status">
                <div class="spinner"></div>
                <span>Launching local services...</span>
              </div>
            </main>
          </body>
        </html>
        """
    ).strip()


def _render_error_html(message: str, log_path: Path) -> str:
    """Return a startup error view for the packaged shell."""
    message_text = html.escape(message)
    log_path_text = html.escape(str(log_path))
    return dedent(
        f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Startup failed</title>
            <style>
              :root {{
                color-scheme: light;
                --bg: #fff7f5;
                --panel: rgba(255, 255, 255, 0.92);
                --text: #3b1d17;
                --muted: #7c5d55;
                --accent: #e85d3f;
                --border: rgba(232, 93, 63, 0.16);
              }}
              * {{
                box-sizing: border-box;
              }}
              body {{
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                padding: 24px;
                font-family: "Segoe UI", "Inter", sans-serif;
                color: var(--text);
                background:
                  radial-gradient(circle at top left, rgba(232, 93, 63, 0.12), transparent 30%),
                  linear-gradient(180deg, #fffdfc 0%, var(--bg) 100%);
              }}
              .shell {{
                width: min(640px, 100%);
                padding: 34px;
                border-radius: 28px;
                background: var(--panel);
                border: 1px solid var(--border);
                box-shadow: 0 24px 60px rgba(59, 29, 23, 0.1);
              }}
              .eyebrow {{
                margin: 0 0 10px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: var(--accent);
              }}
              h1 {{
                margin: 0;
                font-size: 30px;
                line-height: 1.1;
              }}
              p {{
                margin: 14px 0 0;
                font-size: 15px;
                line-height: 1.6;
                color: var(--muted);
              }}
              .error {{
                margin-top: 22px;
                padding: 18px 20px;
                border-radius: 18px;
                background: rgba(232, 93, 63, 0.08);
                color: var(--text);
                font-size: 14px;
                line-height: 1.6;
                white-space: pre-wrap;
              }}
              code {{
                display: block;
                margin-top: 20px;
                padding: 14px 16px;
                border-radius: 14px;
                background: rgba(59, 29, 23, 0.06);
                color: var(--text);
                font-family: "Cascadia Code", "Consolas", monospace;
                font-size: 13px;
                overflow-wrap: anywhere;
              }}
            </style>
          </head>
          <body>
            <main class="shell">
              <p class="eyebrow">Local Document Intelligence</p>
              <h1>Startup needs attention</h1>
              <p>The desktop shell could not start the local backend. You can close this window, then inspect the launcher log below.</p>
              <div class="error">{message_text}</div>
              <code>{log_path_text}</code>
            </main>
          </body>
        </html>
        """
    ).strip()


class BackendProcess:
    """Start and stop the local FastAPI backend for the desktop shell."""

    def __init__(self, paths: AppPaths) -> None:
        self._paths = paths
        self._process: subprocess.Popen[str] | None = None
        self._log_stream = None
        self._server: uvicorn.Server | None = None
        self._server_thread: Thread | None = None
        self._startup_probe_url = HEALTH_URL
        self._open_log_stream()

    def _open_log_stream(self) -> None:
        """Open the persistent launcher log stream for the process lifetime."""
        if self._log_stream is not None:
            return
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self._log_stream = self._paths.launcher_log_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def _append_log(self, message: str) -> None:
        """Append one launcher event to the packaged launcher log."""
        self._open_log_stream()
        assert self._log_stream is not None
        self._log_stream.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        self._log_stream.flush()

    def start(self) -> None:
        """Start the backend process if it is not already running."""
        if self._process is not None and self._process.poll() is None:
            return
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        runtime_config = load_managed_app_config(self._paths.runtime_config_path)
        managed_python = self._paths.managed_venv_dir / "Scripts" / "python.exe"
        packaging_mode = "frozen" if getattr(sys, "frozen", False) else "source"
        self._startup_probe_url = (
            READY_URL if runtime_config.install_state == "ready" else HEALTH_URL
        )
        self._append_log(
            "Launcher startup "
            f"app_root={self._paths.app_root} "
            f"code_root={self._paths.code_root} "
            f"packaging_mode={packaging_mode} "
            f"runtime_config_exists={self._paths.runtime_config_path.is_file()} "
            f"setup_status_exists={self._paths.setup_status_path.is_file()} "
            f"managed_python={managed_python} "
            f"managed_python_exists={managed_python.is_file()} "
            f"runtime_install_state={runtime_config.install_state} "
            f"launcher_log_path={self._paths.launcher_log_path} "
            f"startup_probe_url={self._startup_probe_url}"
        )
        if runtime_config.install_state == "ready" and managed_python.is_file():
            usable, detail = python_executable_is_usable(managed_python)
            if usable:
                self._append_log(
                    "Selected backend runtime_mode=managed_subprocess "
                    f"managed_python={managed_python}"
                )
                self._start_managed_subprocess(managed_python)
                return

            self._append_log(
                "Managed runtime was not usable, falling back to embedded bootstrap "
                f"server. Detail: {detail or 'unknown error'}"
            )

        self._append_log("Selected backend runtime_mode=embedded")
        self._start_embedded_server()

    def _start_embedded_server(self) -> None:
        """Run the bootstrap backend in-process for first-run setup."""
        os.environ[APP_ROOT_ENV_VAR] = str(self._paths.app_root)
        os.environ[CODE_ROOT_ENV_VAR] = str(self._paths.code_root)
        os.environ[BACKEND_RUNTIME_MODE_ENV_VAR] = "embedded"
        os.environ[LAUNCHER_LOG_PATH_ENV_VAR] = str(self._paths.launcher_log_path)
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
        self._open_log_stream()
        assert self._log_stream is not None
        env = sanitized_subprocess_env(
            {
                APP_ROOT_ENV_VAR: str(self._paths.app_root),
                CODE_ROOT_ENV_VAR: str(self._paths.code_root),
                BACKEND_RUNTIME_MODE_ENV_VAR: "managed_subprocess",
                LAUNCHER_LOG_PATH_ENV_VAR: str(self._paths.launcher_log_path),
            }
        )
        self._append_log(
            "Starting backend subprocess "
            f"python={backend_python} cwd={self._paths.code_root}"
        )
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
            **hidden_windows_subprocess_kwargs(),
        )

    def wait_until_ready(self, *, timeout_seconds: int = 90) -> None:
        """Block until the selected backend startup probe is ready."""
        deadline = time.time() + timeout_seconds
        probe_url = self._startup_probe_url
        last_probe_detail = "no readiness response received"
        while time.time() < deadline:
            if self._server_thread is not None and not self._server_thread.is_alive():
                raise RuntimeError("Embedded backend thread exited early.")
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Backend exited early with code {self._process.returncode}."
                )

            try:
                with urllib.request.urlopen(probe_url, timeout=2) as response:
                    if response.status == 200:
                        self._append_log(
                            f"Backend startup probe succeeded url={probe_url}"
                        )
                        return
                    last_probe_detail = (
                        f"unexpected startup probe status={response.status} url={probe_url}"
                    )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").strip()
                last_probe_detail = (
                    f"http_error status={exc.code} url={probe_url} body={body[:800]}"
                )
            except (urllib.error.URLError, TimeoutError) as exc:
                last_probe_detail = f"{type(exc).__name__}: {exc}"

            time.sleep(1)

        self._append_log(
            "Backend startup probe timed out "
            f"url={probe_url} detail={last_probe_detail}"
        )
        raise TimeoutError(
            "Backend did not become ready in time. "
            f"Last startup probe result: {last_probe_detail}"
        )

    def stop(self) -> None:
        """Stop the backend process if it is running."""
        if self._server is not None:
            self._server.should_exit = True
            if self._server_thread is not None:
                self._server_thread.join(timeout=20)
            self._server = None
            self._server_thread = None

        if self._process is not None:
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
    paths.ensure_exists()
    backend = BackendProcess(paths)
    backend._append_log("Launcher main initialized.")
    atexit.register(backend.stop)

    window = webview.create_window(
        title="Starting Local Document Intelligence",
        html=_render_startup_html(),
        width=1480,
        height=980,
        min_size=(1120, 760),
    )

    def bootstrap_shell() -> None:
        try:
            window.load_html(
                _render_startup_html(
                    heading="Starting your local workspace",
                    message="Preparing the local backend and runtime. The full app will open in this window once everything is ready.",
                )
            )
            backend.start()
            backend.wait_until_ready()
            window.set_title("Local Document Intelligence")
            window.load_url(APP_URL)
        except Exception as exc:
            backend.stop()
            window.set_title("Local Document Intelligence - Startup issue")
            window.load_html(_render_error_html(str(exc), paths.launcher_log_path))

    try:
        webview.start(bootstrap_shell, debug=False)
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
