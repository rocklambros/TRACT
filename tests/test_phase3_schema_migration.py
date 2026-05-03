"""Tests for Phase 3 schema migration — reviewer_notes + original_hub_id columns."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tract.crosswalk.schema import SCHEMA_SQL, create_database, get_connection, migrate_schema

OLD_SCHEMA_SQL = """
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
"""


def _create_old_db(db_path: Path) -> None:
    """Create a database with the old schema (no reviewer_notes/original_hub_id)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(OLD_SCHEMA_SQL)
    conn.commit()
    conn.close()


def _get_column_names(db_path: Path, table: str = "assignments") -> list[str]:
    """Get column names for a table."""
    conn = sqlite3.connect(str(db_path))
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    conn.close()
    return cols


class TestMigrateSchemaFreshDB:
    def test_fresh_db_returns_empty_list(self, tmp_path):
        db_path = tmp_path / "fresh.db"
        create_database(db_path)
        result = migrate_schema(db_path)
        assert result == []

    def test_fresh_db_has_both_columns(self, tmp_path):
        db_path = tmp_path / "fresh.db"
        create_database(db_path)
        cols = _get_column_names(db_path)
        assert "reviewer_notes" in cols
        assert "original_hub_id" in cols


class TestMigrateSchemaOldDB:
    def test_old_db_adds_both_columns(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_old_db(db_path)
        cols_before = _get_column_names(db_path)
        assert "reviewer_notes" not in cols_before
        assert "original_hub_id" not in cols_before

        result = migrate_schema(db_path)
        assert "reviewer_notes" in result
        assert "original_hub_id" in result

        cols_after = _get_column_names(db_path)
        assert "reviewer_notes" in cols_after
        assert "original_hub_id" in cols_after

    def test_idempotent_second_call_returns_empty(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_old_db(db_path)
        first = migrate_schema(db_path)
        assert len(first) == 2
        second = migrate_schema(db_path)
        assert second == []

    def test_existing_data_preserved_after_migration(self, tmp_path):
        db_path = tmp_path / "old.db"
        _create_old_db(db_path)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO assignments (control_id, hub_id, confidence, provenance) "
            "VALUES (?, ?, ?, ?)",
            ("ctrl-1", "hub-1", 0.85, "active_learning_round_2"),
        )
        conn.execute(
            "INSERT INTO assignments (control_id, hub_id, confidence, provenance) "
            "VALUES (?, ?, ?, ?)",
            ("ctrl-2", "hub-2", 0.92, "ground_truth_T1-AI"),
        )
        conn.commit()
        conn.close()

        migrate_schema(db_path)

        conn = get_connection(db_path)
        rows = conn.execute(
            "SELECT control_id, hub_id, confidence, provenance, reviewer_notes, original_hub_id "
            "FROM assignments ORDER BY control_id"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0]["control_id"] == "ctrl-1"
        assert rows[0]["hub_id"] == "hub-1"
        assert pytest.approx(rows[0]["confidence"]) == 0.85
        assert rows[0]["provenance"] == "active_learning_round_2"
        assert rows[0]["reviewer_notes"] is None
        assert rows[0]["original_hub_id"] is None

        assert rows[1]["control_id"] == "ctrl-2"
        assert rows[1]["confidence"] == pytest.approx(0.92)
        assert rows[1]["provenance"] == "ground_truth_T1-AI"
