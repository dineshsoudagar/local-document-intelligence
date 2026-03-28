"""Helpers for validating Python runtimes used by the app."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def python_executable_is_usable(
    python_path: Path,
    *,
    timeout_seconds: int = 15,
) -> tuple[bool, str | None]:
    """Return whether one Python executable can start successfully."""
    if not python_path.is_file():
        return False, f"Python executable not found: {python_path}"

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    try:
        result = subprocess.run(
            [str(python_path), "-c", "import sys"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
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
