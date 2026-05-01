"""JSON extractor for framework preparation.

Handles three cases:
1. Already a valid FrameworkOutput — passthrough.
2. Top-level JSON array of objects — map fields heuristically.
3. Object with a "controls"/"items"/"data" array — extract from that key.

Public API:
    JsonExtractor.extract(path) -> list[Control]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tract.schema import Control, FrameworkOutput

logger = logging.getLogger(__name__)

_ID_KEYS: tuple[str, ...] = ("control_id", "id", "section_id", "control id")
_TITLE_KEYS: tuple[str, ...] = ("title", "name", "control_name", "control name")
_DESC_KEYS: tuple[str, ...] = ("description", "desc", "text", "body")
_FULL_TEXT_KEYS: tuple[str, ...] = ("full_text", "fulltext", "full text")
_ARRAY_KEYS: tuple[str, ...] = ("controls", "items", "data")


def _find_key(obj: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    """Find the first matching key in obj (case-insensitive)."""
    lower_map = {k.lower(): k for k in obj.keys()}
    for candidate in candidates:
        real_key = lower_map.get(candidate.lower())
        if real_key is not None:
            return real_key
    return None


def _map_object_to_control(obj: dict[str, Any], index: int) -> Control:
    """Map a generic dict to a Control using heuristic key matching."""
    id_key = _find_key(obj, _ID_KEYS)
    title_key = _find_key(obj, _TITLE_KEYS)
    desc_key = _find_key(obj, _DESC_KEYS)
    full_text_key = _find_key(obj, _FULL_TEXT_KEYS)

    control_id = str(obj[id_key]).strip() if id_key else f"ITEM-{index:03d}"
    title = str(obj[title_key]).strip() if title_key else ""

    if desc_key:
        description = str(obj[desc_key]).strip()
    elif title:
        description = title
    else:
        raise ValueError(
            f"Object at index {index} has no recognizable description field. "
            f"Keys found: {list(obj.keys())}"
        )

    full_text = str(obj[full_text_key]).strip() if full_text_key and obj.get(full_text_key) else None

    return Control(
        control_id=control_id,
        title=title,
        description=description,
        full_text=full_text,
    )


class JsonExtractor:
    """Extract controls from a JSON file."""

    def extract(self, path: Path) -> list[Control]:
        """Read a JSON file and extract controls.

        Args:
            path: Path to the JSON file.

        Returns:
            List of Control objects extracted from the file.

        Raises:
            ValueError: If no recognizable control structure is found.
            FileNotFoundError: If the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        text = path.read_text(encoding="utf-8")
        data = json.loads(text)

        if isinstance(data, dict):
            try:
                fw = FrameworkOutput.model_validate(data)
                logger.info(
                    "JSON is already FrameworkOutput (%d controls), using passthrough",
                    len(fw.controls),
                )
                return list(fw.controls)
            except ValidationError:
                pass

        if isinstance(data, list):
            return self._extract_from_array(data)

        if isinstance(data, dict):
            for key in _ARRAY_KEYS:
                lower_map = {k.lower(): k for k in data.keys()}
                real_key = lower_map.get(key)
                if real_key and isinstance(data[real_key], list):
                    logger.info("Found control array under key %r", real_key)
                    return self._extract_from_array(data[real_key])

        raise ValueError(
            f"No recognizable control structure found in {path.name}. "
            f"Expected: FrameworkOutput JSON, an array of objects, or an object "
            f"with a 'controls'/'items'/'data' array key. "
            f"Try --llm for complex nested formats."
        )

    def _extract_from_array(self, items: list[Any]) -> list[Control]:
        """Extract controls from a list of dicts."""
        controls: list[Control] = []
        for i, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict item at index %d: %r", i, type(item))
                continue
            controls.append(_map_object_to_control(item, i))
        logger.info("Extracted %d controls from JSON array", len(controls))
        return controls
