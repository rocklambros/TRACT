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
# neutralize_csv_cell.
CSV_FORMULA_GUARD: Final[str] = "'"


def export_crosswalk(
    db_path: Path,
    output_path: Path,
    fmt: str = "json",
    *,
    framework: str | None = None,
    hub: str | None = None,
    min_confidence: float | None = None,
    status: str | None = None,
) -> Path:
    """Export assignments from the crosswalk database.

    JSON groups by framework name; CSV carries full metadata per row.

    The four filters correspond to `tract export`'s documented flags. They
    previously did not exist: the CLI accepted `--framework`, `--hub`,
    `--min-confidence` and `--status` and passed none of them here, so
    `tract export --framework mitre_atlas` returned every framework and two
    contradictory invocations produced byte-identical files. Silently wrong
    output is worse than a crash, because it gets used.

    `status=None` preserves each format's historical default -- accepted for
    JSON, everything for CSV -- so an existing caller sees no change. Pass
    `status="all"` for no status predicate in either.
    """
    if fmt == "json":
        return _export_json(
            db_path, output_path,
            framework=framework, hub=hub,
            min_confidence=min_confidence,
            status="accepted" if status is None else status,
        )
    elif fmt == "csv":
        return _export_csv(
            db_path, output_path,
            framework=framework, hub=hub,
            min_confidence=min_confidence,
            status="all" if status is None else status,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'json' or 'csv'.")



def _build_filters(
    framework: str | None,
    hub: str | None,
    min_confidence: float | None,
    status: str | None,
) -> tuple[str, list[object]]:
    """SQL predicates and parameters for the documented export filters.

    Shared by both formats, because they previously disagreed: the JSON path
    hardcoded `review_status = 'accepted'` while the CSV path applied no status
    filter at all, so `--status` meant different things depending on `--format`
    and neither honoured what the user asked for.

    `status="all"` means no status predicate. Every filter is parameterised;
    none is interpolated.
    """
    clauses: list[str] = []
    params: list[object] = []
    if framework:
        clauses.append("f.id = ?")
        params.append(framework)
    if hub:
        clauses.append("a.hub_id = ?")
        params.append(hub)
    if min_confidence is not None:
        clauses.append("a.confidence IS NOT NULL AND a.confidence >= ?")
        params.append(min_confidence)
    if status and status != "all":
        clauses.append("a.review_status = ?")
        params.append(status)
    return (" AND ".join(clauses), params)


def _export_json(
    db_path: Path,
    output_path: Path,
    *,
    framework: str | None = None,
    hub: str | None = None,
    min_confidence: float | None = None,
    status: str | None = "accepted",
) -> Path:
    """Export assignments as JSON grouped by framework name."""
    where, params = _build_filters(framework, hub, min_confidence, status)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, a.hub_id, a.confidence, a.provenance, "
            "f.name AS framework_name "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            + (f"WHERE {where} " if where else "")
            + "ORDER BY f.name, a.control_id, a.hub_id",
            params,
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


def neutralize_csv_cell(value: object) -> object:
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


def _export_csv(
    db_path: Path,
    output_path: Path,
    *,
    framework: str | None = None,
    hub: str | None = None,
    min_confidence: float | None = None,
    status: str | None = "all",
) -> Path:
    """Export assignments as CSV with full metadata."""
    where, params = _build_filters(framework, hub, min_confidence, status)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, f.name AS framework, a.hub_id, "
            "a.confidence, a.provenance, a.review_status, "
            "a.reviewer, a.review_date "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            + (f"WHERE {where} " if where else "")
            + "ORDER BY f.name, a.control_id",
            params,
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
                    {k: neutralize_csv_cell(v) for k, v in dict(row).items()}
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
