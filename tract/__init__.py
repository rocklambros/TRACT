"""TRACT — Translating Requirements Across CRE Trees."""
from __future__ import annotations

import os as _os
import subprocess
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

# Force transformers to the PyTorch backend before it (or sentence-transformers)
# is ever imported. TRACT is torch-only. If TensorFlow is also installed,
# transformers auto-imports it during model loading; a broken/conflicting TF
# native library deadlocks on an abseil mutex (tensorflow pywrap preload_check)
# on macOS, wedging `import sentence_transformers`. setdefault respects an
# explicit operator override (USE_TF=1).
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")


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
