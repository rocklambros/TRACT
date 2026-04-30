"""Tests for crosswalk database snapshots."""
from __future__ import annotations

import pytest


@pytest.fixture
def populated_db(tmp_path):
    from tract.crosswalk.schema import create_database
    from tract.crosswalk.store import insert_frameworks, insert_hubs

    db_path = tmp_path / "test.db"
    create_database(db_path)
    insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
    insert_hubs(db_path, [{"id": "h1", "name": "Hub 1", "path": "R", "parent_id": None}])
    return db_path


class TestSnapshot:
    def test_creates_snapshot_record(self, populated_db) -> None:
        from tract.crosswalk.snapshot import take_snapshot

        snap = take_snapshot(populated_db, round_number=0, description="initial")
        assert snap["round_number"] == 0
        assert len(snap["db_hash"]) == 64  # SHA-256 hex

    def test_consecutive_snapshots_differ_after_change(self, populated_db) -> None:
        from tract.crosswalk.snapshot import take_snapshot
        from tract.crosswalk.store import insert_hubs

        snap1 = take_snapshot(populated_db, round_number=0, description="before")
        insert_hubs(populated_db, [{"id": "h2", "name": "Hub 2", "path": "R", "parent_id": None}])
        snap2 = take_snapshot(populated_db, round_number=1, description="after")

        assert snap1["db_hash"] != snap2["db_hash"]

    def test_identical_state_produces_same_hash(self, populated_db) -> None:
        from tract.crosswalk.snapshot import compute_db_hash

        hash1 = compute_db_hash(populated_db)
        hash2 = compute_db_hash(populated_db)
        assert hash1 == hash2
