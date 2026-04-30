"""Tests for crosswalk database schema."""
from __future__ import annotations

import sqlite3

import pytest


class TestCreateSchema:
    def test_creates_all_tables(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "frameworks" in tables
        assert "controls" in tables
        assert "hubs" in tables
        assert "assignments" in tables
        assert "snapshots" in tables

    def test_wal_mode_enabled(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()

        assert mode == "wal"

    def test_foreign_keys_enabled_via_get_connection(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database, get_connection

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = get_connection(db_path)
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        conn.close()

        assert fk == 1

    def test_assignments_has_expected_columns(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(assignments)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id", "control_id", "hub_id", "confidence",
            "in_conformal_set", "is_ood", "provenance",
            "source_link_id", "model_version", "review_status",
            "reviewer", "review_date", "created_at",
        }
        assert expected.issubset(columns)

    def test_idempotent_creation(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)
        create_database(db_path)  # Should not raise
