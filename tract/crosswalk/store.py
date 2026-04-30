"""Crosswalk database CRUD operations.

All write operations use explicit transactions for atomicity.
Read operations use Row factory for dict-like access.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


def insert_hubs(db_path: Path, hubs: list[dict]) -> int:
    """Insert hub records. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
                [(h["id"], h["name"], h["path"], h["parent_id"]) for h in hubs],
            )
        return len(hubs)
    finally:
        conn.close()


def get_hub(db_path: Path, hub_id: str) -> dict | None:
    """Get a single hub by ID."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM hubs WHERE id = ?", (hub_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def count_hubs(db_path: Path) -> int:
    """Count total hubs."""
    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM hubs").fetchone()[0]
    finally:
        conn.close()


def insert_frameworks(db_path: Path, frameworks: list[dict]) -> int:
    """Insert framework records. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
                "VALUES (?, ?, ?, ?, ?)",
                [(f["id"], f["name"], f["version"], f["fetch_date"], f["control_count"]) for f in frameworks],
            )
        return len(frameworks)
    finally:
        conn.close()


def count_frameworks(db_path: Path) -> int:
    """Count total frameworks."""
    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM frameworks").fetchone()[0]
    finally:
        conn.close()


def insert_controls(db_path: Path, controls: list[dict]) -> int:
    """Insert control records. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO controls (id, framework_id, section_id, title, description, full_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(c["id"], c["framework_id"], c["section_id"], c["title"], c["description"], c["full_text"]) for c in controls],
            )
        return len(controls)
    finally:
        conn.close()


def get_controls_by_framework(db_path: Path, framework_id: str) -> list[dict]:
    """Get all controls for a framework."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM controls WHERE framework_id = ?", (framework_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def insert_assignments(db_path: Path, assignments: list[dict]) -> int:
    """Insert assignment records atomically. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO assignments "
                "(control_id, hub_id, confidence, in_conformal_set, is_ood, "
                "provenance, source_link_id, model_version, review_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (a["control_id"], a["hub_id"], a["confidence"],
                     a["in_conformal_set"], a["is_ood"], a["provenance"],
                     a["source_link_id"], a["model_version"], a["review_status"])
                    for a in assignments
                ],
            )
        return len(assignments)
    finally:
        conn.close()


def get_assignments_by_control(db_path: Path, control_id: str) -> list[dict]:
    """Get all assignments for a control."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM assignments WHERE control_id = ?", (control_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_assignments_by_provenance(db_path: Path, provenance: str) -> list[dict]:
    """Get all assignments with a given provenance."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM assignments WHERE provenance = ?", (provenance,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_assignments_by_status(db_path: Path, status: str) -> list[dict]:
    """Get all assignments with a given review status."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM assignments WHERE review_status = ?", (status,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_review_status(
    db_path: Path,
    assignment_id: int,
    status: str,
    reviewer: str | None = None,
    corrected_hub_id: str | None = None,
) -> None:
    """Update review status of an assignment.

    If status is 'corrected' and corrected_hub_id is provided,
    inserts a new assignment with the corrected hub.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE assignments SET review_status = ?, reviewer = ?, review_date = ? "
                "WHERE id = ?",
                (status, reviewer, now, assignment_id),
            )
            if status == "corrected" and corrected_hub_id:
                row = conn.execute(
                    "SELECT * FROM assignments WHERE id = ?", (assignment_id,)
                ).fetchone()
                if row:
                    conn.execute(
                        "INSERT INTO assignments "
                        "(control_id, hub_id, confidence, in_conformal_set, is_ood, "
                        "provenance, source_link_id, model_version, review_status, "
                        "reviewer, review_date) "
                        "VALUES (?, ?, NULL, NULL, 0, ?, NULL, ?, 'accepted', ?, ?)",
                        (row["control_id"], corrected_hub_id,
                         row["provenance"] + "_corrected", row["model_version"],
                         reviewer, now),
                    )
    finally:
        conn.close()
