"""TRACT — Translating Requirements Across CRE Trees."""
from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


try:
    __version__ = _pkg_version("tract")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = _git_short_sha() or "0.0.0+unknown"
