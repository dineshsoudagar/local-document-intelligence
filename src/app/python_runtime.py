"""Helpers for validating Python runtimes and subprocess environments."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_BLOCKED_PYTHON_ENV_KEYS = {
    "PYTHONEXECUTABLE",
    "PYTHONHOME",
    "PYTHONPATH",
}


def _is_blocked_python_env_key(key: str) -> bool:
    """Return whether one environment key should not leak into child Python processes."""
    normalized = key.upper()
    return (
        normalized in _BLOCKED_PYTHON_ENV_KEYS
        or normalized.startswith("_PYI_")
        or normalized.startswith("PYINSTALLER_")
    )


def sanitized_subprocess_env(
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a subprocess environment without inherited bundled-Python state."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not _is_blocked_python_env_key(key)
    }

    if extra_env is not None:
        for key, value in extra_env.items():
            if _is_blocked_python_env_key(key):
                continue
            env[key] = value

    return env


def hidden_windows_subprocess_kwargs() -> dict[str, Any]:
    """Return subprocess kwargs that suppress console windows on Windows."""
    if os.name != "nt":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


def python_executable_is_usable(
    python_path: Path,
    *,
    timeout_seconds: int = 15,
) -> tuple[bool, str | None]:
    """Return whether one Python executable can start successfully."""
    if not python_path.is_file():
        return False, f"Python executable not found: {python_path}"

    try:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                "import socket, sqlite3, ssl, sys",
            ],
            env=sanitized_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            **hidden_windows_subprocess_kwargs(),
        )
    except OSError as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired:
        return False, f"Timed out while starting Python: {python_path}"

    if result.returncode == 0:
        return True, None

    detail = (result.stderr or result.stdout or "").strip()
    if detail:
        return False, detail

    return False, f"Python exited with code {result.returncode}: {python_path}"
