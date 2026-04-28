"""TRACT atomic JSON I/O utilities.

Provides deterministic, crash-safe JSON read/write operations.
All writes go to a temporary file first, then atomically replace
the target via os.replace() — no partial writes on crash.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(data: Any, path: Path | str) -> None:
    """Atomically write data as formatted JSON to *path*.

    Creates parent directories if they don't exist. Writes to a temporary
    file in the same directory, then uses os.replace() for an atomic swap.
    Output is deterministic: sorted keys, 2-space indent, no ASCII escaping,
    trailing newline.

    Args:
        data: Any JSON-serializable Python object.
        path: Destination file path.

    Raises:
        TypeError: If *data* is not JSON-serializable.
        OSError: If the write or rename fails.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory so os.replace() is atomic
    # (same filesystem guarantees atomic rename on POSIX).
    fd, tmp_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_json(path: Path | str) -> Any:
    """Load and return parsed JSON from *path*.

    Args:
        path: File path to read.

    Returns:
        The parsed JSON data (dict, list, etc.).

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
