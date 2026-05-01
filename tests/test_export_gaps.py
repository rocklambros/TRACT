"""Tests for coverage gaps report."""
from __future__ import annotations

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)
from tract.export.gaps import query_coverage_gaps


@pytest.fixture
def gaps_db(tmp_path):
    db_path = tmp_path / "gaps_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-05-01", "control_count": 5},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "C1", "title": "Exported Control", "description": "D1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "C2", "title": "Below Floor", "description": "D2", "full_text": None},
        {"id": "fw1:c3", "framework_id": "fw1", "section_id": "C3", "title": "No Assignment", "description": "D3", "full_text": None},
        {"id": "fw1:c4", "framework_id": "fw1", "section_id": "C4", "title": "Ground Truth", "description": "D4", "full_text": None},
        {"id": "fw1:c5", "framework_id": "fw1", "section_id": "C5", "title": "OOD Control", "description": "D5", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.8, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw1:c2", "hub_id": "h2", "confidence": 0.15, "in_conformal_set": 0, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw1:c4", "hub_id": "h1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "ground_truth_T1-AI", "source_link_id": None, "model_version": "v1", "review_status": "ground_truth"},
        {"control_id": "fw1:c5", "hub_id": "h1", "confidence": 0.7, "in_conformal_set": 1, "is_ood": 1,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
    ])
    return db_path


class TestCoverageGaps:
    def test_identifies_all_gap_reasons(self, gaps_db) -> None:
        exported_keys = {("fw1:c1", "h1")}
        gaps = query_coverage_gaps(
            gaps_db, exported_keys=exported_keys,
            confidence_floor=0.30, confidence_overrides={},
            framework_ids=["fw1"],
        )
        fw1 = gaps["fw1"]
        assert fw1["total_controls"] == 5
        assert fw1["exported_controls"] == 1
        missing = {m["section_id"]: m for m in fw1["missing_controls"]}
        assert len(missing) == 4
        assert missing["C2"]["reason"] == "below_confidence_floor"
        assert missing["C3"]["reason"] == "no_assignment"
        assert missing["C4"]["reason"] == "ground_truth"
        assert missing["C5"]["reason"] == "out_of_distribution"

    def test_coverage_pct(self, gaps_db) -> None:
        exported_keys = {("fw1:c1", "h1")}
        gaps = query_coverage_gaps(
            gaps_db, exported_keys=exported_keys,
            confidence_floor=0.30, confidence_overrides={},
            framework_ids=["fw1"],
        )
        assert gaps["fw1"]["coverage_pct"] == 20.0

    def test_below_floor_includes_confidence(self, gaps_db) -> None:
        exported_keys = {("fw1:c1", "h1")}
        gaps = query_coverage_gaps(
            gaps_db, exported_keys=exported_keys,
            confidence_floor=0.30, confidence_overrides={},
            framework_ids=["fw1"],
        )
        c2 = [m for m in gaps["fw1"]["missing_controls"] if m["section_id"] == "C2"][0]
        assert "0.150" in c2["detail"]
        assert "0.30" in c2["detail"]

    def test_override_changes_floor(self, gaps_db) -> None:
        exported_keys = {("fw1:c1", "h1")}
        gaps = query_coverage_gaps(
            gaps_db, exported_keys=exported_keys,
            confidence_floor=0.10, confidence_overrides={"fw1": 0.50},
            framework_ids=["fw1"],
        )
        c2 = [m for m in gaps["fw1"]["missing_controls"] if m["section_id"] == "C2"][0]
        assert c2["reason"] == "below_confidence_floor"
        assert "0.50" in c2["detail"]

    def test_empty_framework(self, gaps_db) -> None:
        gaps = query_coverage_gaps(
            gaps_db, exported_keys=set(),
            confidence_floor=0.30, confidence_overrides={},
            framework_ids=["nonexistent"],
        )
        assert gaps["nonexistent"]["total_controls"] == 0
        assert gaps["nonexistent"]["missing_controls"] == []
