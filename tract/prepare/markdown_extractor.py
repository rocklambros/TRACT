"""Markdown extractor for framework preparation.

Splits markdown documents on heading patterns. Each heading becomes a
control: heading text -> title, body text -> description, ID extracted
from heading prefix patterns or generated positionally.

Public API:
    MarkdownExtractor.extract(path) -> list[Control]
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tract.schema import Control

logger = logging.getLogger(__name__)

_ID_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^([A-Za-z]{2,10}\d+(?:\.\d+)*)\s*[:\-–—]\s*(.+)$"),
    re.compile(r"^([A-Za-z]+-\d+(?:\.\d+)*)\s*[:\-–—]\s*(.+)$"),
    re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$"),
    re.compile(r"^(\d+\.\d+)\s+(.+)$"),
]

_HEADING_RE: re.Pattern[str] = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _parse_heading_id(text: str) -> tuple[str | None, str]:
    """Extract a control ID and clean title from heading text.

    Returns:
        (control_id, title) — control_id is None if no pattern matched.
    """
    text = text.strip()
    for pattern in _ID_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1), match.group(2).strip()
    return None, text


def _detect_heading_level(text: str) -> int:
    """Auto-detect the heading level that has the most body text."""
    level_counts: dict[int, int] = {}
    lines = text.split("\n")

    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            has_body = False
            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith("#"):
                    has_body = True
                    break
                if next_line.startswith("#"):
                    break
            if has_body:
                level_counts[level] = level_counts.get(level, 0) + 1

    if not level_counts:
        return 2

    best_level = max(level_counts, key=lambda k: level_counts[k])
    logger.info("Auto-detected heading level %d (counts: %s)", best_level, level_counts)
    return best_level


class MarkdownExtractor:
    """Extract controls from a markdown document by splitting on headings."""

    def __init__(self, heading_level: int | None = None) -> None:
        self._heading_level = heading_level

    def extract(self, path: Path) -> list[Control]:
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        text = path.read_text(encoding="utf-8")
        level = self._heading_level or _detect_heading_level(text)

        sections: list[tuple[str, str]] = []
        lines = text.split("\n")
        current_heading: str | None = None
        body_lines: list[str] = []

        for line in lines:
            match = _HEADING_RE.match(line)
            if match and len(match.group(1)) == level:
                if current_heading is not None:
                    sections.append((current_heading, "\n".join(body_lines).strip()))
                current_heading = match.group(2).strip()
                body_lines = []
            elif match and len(match.group(1)) < level:
                if current_heading is not None:
                    sections.append((current_heading, "\n".join(body_lines).strip()))
                current_heading = None
                body_lines = []
            elif current_heading is not None:
                body_lines.append(line)

        if current_heading is not None:
            sections.append((current_heading, "\n".join(body_lines).strip()))

        if not sections:
            raise ValueError(
                f"No extractable headings found at level {level} in {path.name}. Options:\n"
                "  (1) Restructure with ## headings per control\n"
                "  (2) Convert to CSV with id/title/description columns\n"
                "  (3) Use --llm with ANTHROPIC_API_KEY env var for LLM extraction"
            )

        controls: list[Control] = []
        positional_counter = 0

        for heading_text, body_text in sections:
            if not body_text:
                logger.debug("Skipping heading with no body: %r", heading_text)
                continue

            control_id, title = _parse_heading_id(heading_text)
            if control_id is None:
                positional_counter += 1
                control_id = f"CTRL-{positional_counter:02d}"

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=body_text,
            ))

        if not controls:
            raise ValueError(
                f"No controls with body text found at heading level {level} "
                f"in {path.name}."
            )

        logger.info("Extracted %d controls from markdown %s (heading level %d)", len(controls), path.name, level)
        return controls
