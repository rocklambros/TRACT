"""Crosswalk database schema and initialization.

SQLite database with 5 tables: frameworks, controls, hubs, assignments, snapshots.
Uses WAL mode for concurrent read access. All tables use IF NOT EXISTS for
idempotent creation.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frameworks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT,
    fetch_date TEXT,
    control_count INTEGER
);

CREATE TABLE IF NOT EXISTS controls (
    id TEXT PRIMARY KEY,
    framework_id TEXT NOT NULL REFERENCES frameworks(id),
    section_id TEXT NOT NULL,
    title TEXT,
    description TEXT,
    full_text TEXT
);

CREATE TABLE IF NOT EXISTS hubs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT,
    parent_id TEXT REFERENCES hubs(id)
);

CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    control_id TEXT NOT NULL REFERENCES controls(id),
    hub_id TEXT NOT NULL REFERENCES hubs(id),
    confidence REAL,
    in_conformal_set INTEGER,
    is_ood INTEGER DEFAULT 0,
    provenance TEXT NOT NULL,
    source_link_id TEXT,
    model_version TEXT,
    review_status TEXT DEFAULT 'pending',
    reviewer TEXT,
    review_date TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number INTEGER NOT NULL,
    snapshot_date TEXT DEFAULT (datetime('now')),
    db_hash TEXT NOT NULL,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_assignments_control ON assignments(control_id);
CREATE INDEX IF NOT EXISTS idx_assignments_hub ON assignments(hub_id);
CREATE INDEX IF NOT EXISTS idx_assignments_provenance ON assignments(provenance);
CREATE INDEX IF NOT EXISTS idx_assignments_review ON assignments(review_status);
"""


def create_database(db_path: Path) -> None:
    """Create the crosswalk database with all tables and indices.

    Idempotent — safe to call on an existing database.
    Enables WAL mode and foreign key enforcement.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Crosswalk database initialized at %s", db_path)
    finally:
        conn.close()


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
