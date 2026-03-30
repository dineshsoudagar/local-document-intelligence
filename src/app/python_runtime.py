"""Helpers for validating Python runtimes and subprocess environments."""

from __future__ import annotations

import os
import shutil
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


def ensure_embedded_python_runtime(
    *,
    bundled_python_dir: Path,
    embedded_python_dir: Path,
) -> Path:
    """Persist the packaged bootstrap Python runtime into app storage."""
    embedded_python = embedded_python_dir / "python.exe"
    usable, _ = python_executable_is_usable(embedded_python)
    if usable:
        return embedded_python

    bundled_python = bundled_python_dir / "python.exe"
    if not bundled_python.is_file():
        return embedded_python

    if embedded_python_dir.exists():
        shutil.rmtree(embedded_python_dir)

    shutil.copytree(bundled_python_dir, embedded_python_dir)
    return embedded_python


def sync_managed_venv_base_paths(
    *,
    managed_venv_dir: Path,
    base_python_dir: Path,
) -> bool:
    """Point one managed venv at a stable bootstrap Python runtime."""
    pyvenv_cfg_path = managed_venv_dir / "pyvenv.cfg"
    base_python = base_python_dir / "python.exe"
    if not pyvenv_cfg_path.is_file() or not base_python.is_file():
        return False

    desired_values = {
        "home": str(base_python_dir.resolve()),
        "executable": str(base_python.resolve()),
        "command": f"{base_python.resolve()} -m venv {managed_venv_dir.resolve()}",
    }
    seen_keys: set[str] = set()
    changed = False
    updated_lines: list[str] = []

    for line in pyvenv_cfg_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        matched = False
        for key, value in desired_values.items():
            prefix = f"{key} = "
            if stripped.startswith(prefix):
                new_line = f"{key} = {value}"
                seen_keys.add(key)
                if line != new_line:
                    changed = True
                updated_lines.append(new_line)
                matched = True
                break
        if not matched:
            updated_lines.append(line)

    for key, value in desired_values.items():
        if key in seen_keys:
            continue
        changed = True
        updated_lines.append(f"{key} = {value}")

    if changed:
        pyvenv_cfg_path.write_text(
            "\n".join(updated_lines) + "\n",
            encoding="utf-8",
        )

    return True


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
