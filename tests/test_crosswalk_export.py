"""Tests for crosswalk export."""
from __future__ import annotations

import csv
import json

import pytest


@pytest.fixture
def populated_db(tmp_path):
    from tract.crosswalk.schema import create_database
    from tract.crosswalk.store import (
        insert_assignments,
        insert_controls,
        insert_frameworks,
        insert_hubs,
        update_review_status,
    )

    db_path = tmp_path / "test.db"
    create_database(db_path)
    insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2}])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": "Desc1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2", "title": "C2", "description": "Desc2", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0, "provenance": "training_T1", "source_link_id": "link1", "model_version": "v1", "review_status": "pending"},
        {"control_id": "fw1:c2", "hub_id": "h2", "confidence": 0.7, "in_conformal_set": 0, "is_ood": 0, "provenance": "al_r1", "source_link_id": None, "model_version": "v1", "review_status": "pending"},
    ])
    update_review_status(db_path, 1, "accepted", reviewer="expert")
    return db_path


class TestExportJSON:
    def test_exports_accepted_only(self, populated_db, tmp_path) -> None:
        from tract.crosswalk.export import export_crosswalk

        out = export_crosswalk(populated_db, tmp_path / "out.json", fmt="json")
        data = json.loads(out.read_text(encoding="utf-8"))

        assert "FW1" in data
        assert len(data["FW1"]) == 1
        assert "fw1:c1" in data["FW1"]


class TestExportCSV:
    def test_exports_all_with_status(self, populated_db, tmp_path) -> None:
        from tract.crosswalk.export import export_crosswalk

        out = export_crosswalk(populated_db, tmp_path / "out.csv", fmt="csv")
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["review_status"] in ("accepted", "pending")
