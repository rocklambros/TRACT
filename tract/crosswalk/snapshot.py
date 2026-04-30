"""Pre-round database snapshots for crosswalk integrity tracking."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)

SNAPSHOT_TABLES = ["frameworks", "controls", "hubs", "assignments"]


def compute_db_hash(db_path: Path) -> str:
    """Compute SHA-256 hash of all data in snapshot-tracked tables.

    Reads all rows from each table in deterministic order and hashes
    the JSON-serialized result.
    """
    conn = get_connection(db_path)
    try:
        state: dict[str, list[list]] = {}
        for table in SNAPSHOT_TABLES:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY rowid"  # noqa: S608
            ).fetchall()
            state[table] = [list(row) for row in rows]

        canonical = json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    finally:
        conn.close()


def take_snapshot(db_path: Path, round_number: int, description: str) -> dict:
    """Record a snapshot of the current database state.

    Returns the snapshot record as a dict.
    """
    db_hash = compute_db_hash(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "INSERT INTO snapshots (round_number, db_hash, description) "
                "VALUES (?, ?, ?)",
                (round_number, db_hash, description),
            )
            snap_id = cursor.lastrowid

        row = conn.execute(
            "SELECT * FROM snapshots WHERE id = ?", (snap_id,)
        ).fetchone()
        result = dict(row)
        logger.info(
            "Snapshot %d: round=%d, hash=%s, desc=%s",
            snap_id, round_number, db_hash[:16], description,
        )
        return result
    finally:
        conn.close()
