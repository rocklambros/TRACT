"""Tests for canonical export snapshot builder and differ (spec §§2-7)."""
from __future__ import annotations

import json

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def canonical_db(tmp_path):
    """DB with representative data for canonical export tests."""
    db_path = tmp_path / "canonical_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-05-04", "control_count": 3},
        {"id": "fw2", "name": "FW2", "version": "1.0", "fetch_date": "2026-05-04", "control_count": 1},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1",
         "title": "Control 1", "description": "Desc 1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2",
         "title": "Control 2", "description": "Desc 2", "full_text": None},
        {"id": "fw1:c3", "framework_id": "fw1", "section_id": "c3",
         "title": "Control 3", "description": "Desc 3", "full_text": None},
        {"id": "fw2:c1", "framework_id": "fw2", "section_id": "c1",
         "title": "FW2 Control 1", "description": "FW2 Desc 1", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.8,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw1:c2", "hub_id": "h2", "confidence": 0.6,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw1:c3", "hub_id": "h1", "confidence": 0.2,
         "in_conformal_set": 0, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw2:c1", "hub_id": "h2", "confidence": 0.7,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
    ])
    return db_path


class TestBuildSnapshot:
    def test_builds_valid_snapshot(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name="FW1",
        )
        assert snap.framework_id == "fw1"
        assert snap.model_adapter_hash == "abc123"
        assert len(snap.controls) == 2  # c3 filtered by 0.3 floor
        assert len(snap.mappings) == 2
        assert snap.content_hash != "placeholder"
        assert len(snap.content_hash) == 64

    def test_all_mappings_get_model_version(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name="FW1",
        )
        for m in snap.mappings:
            assert m.model_version == "abc123"

    def test_rank_is_one_for_single_hub(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name="FW1",
        )
        for m in snap.mappings:
            assert m.rank == 1

    def test_filters_by_confidence_floor(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.5,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name="FW1",
        )
        assert len(snap.mappings) == 2
        confidences = [m.confidence for m in snap.mappings]
        assert all(c >= 0.5 for c in confidences)

    def test_hyperlink_populated(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name="FW1",
        )
        for c in snap.controls:
            assert c.hyperlink.startswith("https://example.com/fw1/")
