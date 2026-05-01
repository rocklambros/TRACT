"""Tests for OpenCRE export filter pipeline."""
from __future__ import annotations

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def filter_db(tmp_path):
    db_path = tmp_path / "filter_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 5},
        {"id": "fw2", "name": "FW2", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": "D1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2", "title": "C2", "description": "D2", "full_text": None},
        {"id": "fw1:c3", "framework_id": "fw1", "section_id": "c3", "title": "C3", "description": "D3", "full_text": None},
        {"id": "fw1:c4", "framework_id": "fw1", "section_id": "c4", "title": "C4", "description": "D4", "full_text": None},
        {"id": "fw1:c5", "framework_id": "fw1", "section_id": "c5", "title": "C5", "description": "D5", "full_text": None},
        {"id": "fw2:c1", "framework_id": "fw2", "section_id": "c1", "title": "C1", "description": "D1", "full_text": None},
        {"id": "fw2:c2", "framework_id": "fw2", "section_id": "c2", "title": "C2", "description": "D2", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "ground_truth_T1-AI", "source_link_id": None, "model_version": "v1", "review_status": "ground_truth"},
        {"control_id": "fw1:c2", "hub_id": "h1", "confidence": 0.6, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw1:c3", "hub_id": "h2", "confidence": 0.25, "in_conformal_set": 0, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw1:c4", "hub_id": "h1", "confidence": 0.5, "in_conformal_set": 0, "is_ood": 1,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw1:c5", "hub_id": "h1", "confidence": None, "in_conformal_set": 0, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw2:c1", "hub_id": "h2", "confidence": 0.7, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw2:c2", "hub_id": "h1", "confidence": 0.8, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "pending"},
    ])
    return db_path


class TestFilterPipeline:
    def test_excludes_ground_truth(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c1" not in control_ids

    def test_excludes_null_confidence(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c5" not in control_ids

    def test_excludes_ood(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c4" not in control_ids

    def test_excludes_below_global_floor(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(filter_db, confidence_floor=0.30, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c3" not in control_ids

    def test_keeps_above_global_floor(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(filter_db, confidence_floor=0.30, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c2" in control_ids
        assert "fw2:c1" in control_ids

    def test_excludes_non_accepted(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw2:c2" not in control_ids

    def test_per_framework_override(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(
            filter_db, confidence_floor=0.30, confidence_overrides={"fw1": 0.65},
        )
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c2" not in control_ids
        assert "fw2:c1" in control_ids

    def test_returns_required_columns(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        assert len(rows) > 0
        required_keys = {"control_id", "hub_id", "hub_name", "confidence",
                         "framework_id", "section_id", "title", "description"}
        assert required_keys.issubset(set(rows[0].keys()))

    def test_filter_stats_counts(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments, compute_filter_stats
        rows = query_exportable_assignments(filter_db, confidence_floor=0.30, confidence_overrides={})
        stats = compute_filter_stats(filter_db, rows, confidence_floor=0.30, confidence_overrides={})
        assert stats["fw1"]["exported"] == 1
        assert stats["fw1"]["excluded_ground_truth"] == 1
        assert stats["fw1"]["excluded_confidence"] >= 1
        assert stats["fw1"]["excluded_ood"] == 1

    def test_framework_filter(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments
        rows = query_exportable_assignments(
            filter_db, confidence_floor=0.0, confidence_overrides={}, framework_filter="fw2",
        )
        framework_ids = {r["framework_id"] for r in rows}
        assert framework_ids == {"fw2"}
