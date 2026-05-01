"""Format detection and extractor registry for framework preparation.

Public API:
    detect_format(path) -> str
    ExtractorRegistry — maps format strings to extractor classes
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from tract.schema import Control

logger = logging.getLogger(__name__)

_EXTENSION_FORMAT_MAP: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".pdf": "unstructured",
    ".html": "unstructured",
    ".htm": "unstructured",
    ".txt": "unstructured",
    ".docx": "unstructured",
}


class Extractor(Protocol):
    """Protocol for format-specific control extractors."""

    def extract(self, path: Path) -> list[Control]: ...


def detect_format(path: Path) -> str:
    """Detect the document format from file extension.

    Returns:
        One of "csv", "markdown", "json", "unstructured".

    Raises:
        ValueError: If the extension is not recognized.
    """
    suffix = path.suffix.lower()
    fmt = _EXTENSION_FORMAT_MAP.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Unrecognized file extension: {suffix!r}. "
            f"Use --format to specify one of: csv, markdown, json, unstructured"
        )
    logger.info("Detected format %r for %s (extension: %s)", fmt, path.name, suffix)
    return fmt


class ExtractorRegistry:
    """Registry mapping format names to extractor instances."""

    def __init__(self) -> None:
        self._extractors: dict[str, Extractor] = {}

    def register(self, format_name: str, extractor: Extractor) -> None:
        """Register an extractor for a format name."""
        self._extractors[format_name] = extractor

    def get(self, format_name: str) -> Extractor:
        """Get the extractor for a format name.

        Raises:
            ValueError: If no extractor is registered for the format.
        """
        extractor = self._extractors.get(format_name)
        if extractor is None:
            registered = ", ".join(sorted(self._extractors.keys()))
            raise ValueError(
                f"No extractor registered for format {format_name!r}. "
                f"Available: {registered}"
            )
        return extractor

    @property
    def formats(self) -> list[str]:
        """Return sorted list of registered format names."""
        return sorted(self._extractors.keys())
