"""CSV/TSV extractor for framework preparation.

Public API:
    CsvExtractor.extract(path) -> list[Control]
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from tract.schema import Control

logger = logging.getLogger(__name__)

_COLUMN_ALIASES: dict[str, str] = {
    "control_id": "control_id",
    "id": "control_id",
    "control id": "control_id",
    "section_id": "control_id",
    "section id": "control_id",
    "title": "title",
    "name": "title",
    "control_name": "title",
    "control name": "title",
    "description": "description",
    "desc": "description",
    "text": "description",
    "body": "description",
    "full_text": "full_text",
    "fulltext": "full_text",
    "full text": "full_text",
}


def _resolve_columns(header: list[str]) -> dict[str, str]:
    """Map CSV header columns to canonical Control field names.

    Returns:
        Dict mapping canonical field name -> CSV column name.

    Raises:
        ValueError: If required columns (control_id and description) cannot
            be identified.
    """
    resolved: dict[str, str] = {}
    for col in header:
        normalized = col.strip().lower()
        canonical = _COLUMN_ALIASES.get(normalized)
        if canonical and canonical not in resolved:
            resolved[canonical] = col

    missing = []
    if "control_id" not in resolved:
        missing.append("control_id (or: id, section_id)")
    if "description" not in resolved:
        missing.append("description (or: desc, text, body)")

    if missing:
        raise ValueError(
            f"CSV missing required column(s): {', '.join(missing)}. "
            f"Found columns: {header}. "
            f"Use --id-column / --description-column to specify explicitly."
        )
    return resolved


class CsvExtractor:
    """Extract controls from a CSV or TSV file."""

    def extract(self, path: Path) -> list[Control]:
        """Read a CSV/TSV and convert rows to Control objects.

        Raises:
            ValueError: If required columns are missing.
            FileNotFoundError: If the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

        text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")

        col_map = _resolve_columns(list(reader.fieldnames))
        logger.info("CSV column mapping for %s: %s", path.name, {k: v for k, v in col_map.items()})

        controls: list[Control] = []
        for row_num, row in enumerate(reader, start=2):
            control_id = row.get(col_map["control_id"], "").strip()
            description = row.get(col_map["description"], "").strip()
            title = row.get(col_map.get("title", ""), "").strip() if "title" in col_map else ""
            full_text = (
                row.get(col_map.get("full_text", ""), "").strip()
                if "full_text" in col_map
                else None
            )

            if not control_id and not description and not title:
                logger.debug("Skipping empty row %d in %s", row_num, path.name)
                continue

            if not control_id:
                control_id = f"ROW-{row_num}"
                logger.warning("Row %d in %s has no control_id, using %s", row_num, path.name, control_id)

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=description if description else title,
                full_text=full_text if full_text else None,
            ))

        logger.info("Extracted %d controls from CSV %s", len(controls), path.name)
        return controls
