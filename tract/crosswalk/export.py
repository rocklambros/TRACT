"""Export crosswalk assignments to JSON or CSV."""
from __future__ import annotations

import csv
import json
import logging
from typing import Any, Final
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)

# Excel, LibreOffice and Google Sheets all evaluate a cell whose first
# character is one of these when the file is opened, so a framework name or a
# provenance string that left the database as `=HYPERLINK(...)` or `@SUM(1+1)`
# is executable content by the time an analyst sees it. Tab and carriage return
# are on the list because both are stripped before that decision is made, which
# turns "\t=..." back into "=...". A newline is deliberately NOT on it: an
# embedded newline forces the writer to quote the field, and the cell then
# starts empty, which no spreadsheet parses as a formula.
CSV_FORMULA_TRIGGERS: Final[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")

# A leading apostrophe is the one neutralisation every spreadsheet honours, and
# it IS visible in the cell, so it is spent only where it buys something. Two
# legitimate shapes stay untouched: an identifier like "A-1" never reaches the
# test at all, because the trigger is positional, and a string that parses as a
# number ("-0.5", "-1") is passed through by the carve-out in
# _neutralize_csv_cell.
CSV_FORMULA_GUARD: Final[str] = "'"


def export_crosswalk(db_path: Path, output_path: Path, fmt: str = "json") -> Path:
    """Export assignments from the crosswalk database.

    JSON format exports only accepted assignments grouped by framework.
    CSV format exports all assignments with full metadata.
    """
    if fmt == "json":
        return _export_json(db_path, output_path)
    elif fmt == "csv":
        return _export_csv(db_path, output_path)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'json' or 'csv'.")


def _export_json(db_path: Path, output_path: Path) -> Path:
    """Export accepted assignments as JSON grouped by framework name."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, a.hub_id, a.confidence, a.provenance, "
            "f.name AS framework_name "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            "WHERE a.review_status = 'accepted' "
            "ORDER BY f.name, a.control_id, a.hub_id"
        ).fetchall()
    finally:
        conn.close()

    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        result[row["framework_name"]][row["control_id"]].append({
            "hub_id": row["hub_id"],
            "confidence": row["confidence"],
            "provenance": row["provenance"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Exported %d accepted assignments to %s", len(rows), output_path)
    return output_path


def _neutralize_csv_cell(value: object) -> object:
    """Return *value* with its formula trigger disarmed, if it has one.

    Applied at the CSV boundary and nowhere else. The stored string is the
    framework's own text, and the JSON export, the API and the review flow all
    want it verbatim; only the spreadsheet reading is dangerous, so only the
    spreadsheet writer pays. That also means the neutralisation has to be
    re-applied by any future exporter -- the hostile string is still in the
    database.

    Non-strings come back untouched, which matters rather than being defensive
    padding: every row carries a float `confidence` and a nullable `reviewer`,
    and a guard that indexed value[0] would raise TypeError on the first export
    anyone ran. A string that parses as a number is also passed through, so a
    negative confidence does not acquire an apostrophe. `float()` accepts
    underscores, so "-1_0" is passed through as well; it is a number in a cell
    either way, not a formula.
    """
    if not isinstance(value, str) or not value.startswith(CSV_FORMULA_TRIGGERS):
        return value
    try:
        float(value)
    except ValueError:
        return CSV_FORMULA_GUARD + value
    return value


def _export_csv(db_path: Path, output_path: Path) -> Path:
    """Export all assignments as CSV with full metadata."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, f.name AS framework, a.hub_id, "
            "a.confidence, a.provenance, a.review_status, "
            "a.reviewer, a.review_date "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            "ORDER BY f.name, a.control_id"
        ).fetchall()
    finally:
        conn.close()

    fieldnames = ["control_id", "framework", "hub_id", "confidence",
                  "provenance", "review_status", "reviewer", "review_date"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Every column, not an enumerated subset: which of them hold
                # attacker-influenced text changes whenever the schema does.
                writer.writerow(
                    {k: _neutralize_csv_cell(v) for k, v in dict(row).items()}
                )
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Exported %d assignments to %s", len(rows), output_path)
    return output_path
