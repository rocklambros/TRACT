"""Export crosswalk assignments to JSON or CSV."""
from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


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

    result: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
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
                writer.writerow(dict(row))
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Exported %d assignments to %s", len(rows), output_path)
    return output_path
