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


def _make_control(
    control_id: str = "fw1:c1",
    framework_id: str = "fw1",
    section_id: str = "c1",
    title: str = "Control 1",
    description: str = "Description 1",
    hyperlink: str = "https://example.com",
) -> dict:
    return {
        "control_id": control_id,
        "framework_id": framework_id,
        "section_id": section_id,
        "title": title,
        "description": description,
        "hyperlink": hyperlink,
    }


def _make_mapping(
    control_id: str = "fw1:c1",
    hub_id: str = "004-517",
    hub_name: str = "Security requirements",
    confidence: float = 0.75,
    rank: int = 1,
    provenance: str = "active_learning_round_2",
    model_version: str = "7e8b8f834db5",
) -> dict:
    return {
        "control_id": control_id,
        "hub_id": hub_id,
        "hub_name": hub_name,
        "confidence": confidence,
        "rank": rank,
        "link_type": "TRACT_ML_PREDICTED",
        "provenance": provenance,
        "model_version": model_version,
    }


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


class TestDiffSnapshots:
    def _make_snapshot(self, controls, mappings, framework_id="fw1"):
        from tract.export.canonical_schema import (
            CanonicalControl, CREMapping, FilterPolicy, StandardSnapshot, compute_content_hash,
        )
        snap = StandardSnapshot(
            framework_id=framework_id,
            framework_name="FW1",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="abc123",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**c) for c in controls],
            mappings=[CREMapping(**m) for m in mappings],
        )
        snap.content_hash = compute_content_hash(snap)
        return snap

    def test_initial_export_all_adds(self) -> None:
        from tract.export.canonical import diff_snapshots

        current = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=None, current=current)
        assert cs.from_version is None
        assert len(cs.operations) == 2
        ops = {op.operation for op in cs.operations}
        assert ops == {"ADD_CONTROL", "ADD_MAPPING"}
        assert cs.summary.controls_added == 1
        assert cs.summary.mappings_added == 1

    def test_no_changes_empty_changeset(self) -> None:
        from tract.export.canonical import diff_snapshots

        snap = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=snap, current=snap)
        assert len(cs.operations) == 0
        assert cs.summary.controls_added == 0

    def test_added_control_detected(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        c2 = _make_control(control_id="fw1:c2", section_id="c2", title="New")
        m2 = _make_mapping(control_id="fw1:c2", hub_id="h2")
        current = self._make_snapshot(
            controls=[_make_control(), c2],
            mappings=[_make_mapping(), m2],
        )
        cs = diff_snapshots(prior=prior, current=current)
        add_ops = [op for op in cs.operations if op.operation.startswith("ADD_")]
        assert len(add_ops) == 2

    def test_deleted_control_detected(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        current = self._make_snapshot(controls=[], mappings=[])
        cs = diff_snapshots(prior=prior, current=current)
        del_ops = [op for op in cs.operations if op.operation.startswith("DELETE_")]
        assert len(del_ops) == 2
        assert cs.summary.controls_deleted == 1
        assert cs.summary.mappings_deleted == 1

    def test_updated_control_detected(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        updated_control = _make_control(title="Changed Title")
        current = self._make_snapshot(
            controls=[updated_control],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=prior, current=current)
        update_ops = [op for op in cs.operations if op.operation == "UPDATE_CONTROL"]
        assert len(update_ops) == 1
        assert update_ops[0].before is not None
        assert update_ops[0].entity is not None

    def test_updated_mapping_confidence_change(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping(confidence=0.75)],
        )
        current = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping(confidence=0.85)],
        )
        cs = diff_snapshots(prior=prior, current=current)
        update_ops = [op for op in cs.operations if op.operation == "UPDATE_MAPPING"]
        assert len(update_ops) == 1

    def test_impact_analysis_populated(self) -> None:
        from tract.export.canonical import diff_snapshots

        current = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=None, current=current)
        assert cs.impact.scope == "minor"
        assert "004-517" in cs.impact.affected_hubs


class TestExportHistory:
    def test_ensure_table_idempotent(self, canonical_db) -> None:
        from tract.export.canonical import ensure_export_history_table

        ensure_export_history_table(canonical_db)
        ensure_export_history_table(canonical_db)  # second call is safe

    def test_save_and_load_snapshot(self, canonical_db) -> None:
        from tract.export.canonical import (
            build_snapshot,
            ensure_export_history_table,
            load_prior_snapshot,
            save_to_export_history,
        )

        ensure_export_history_table(canonical_db)
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
        save_to_export_history(canonical_db, snap)
        prior = load_prior_snapshot(canonical_db, "fw1")
        assert prior is not None
        assert prior.content_hash == snap.content_hash
        assert prior.framework_id == "fw1"
        assert len(prior.controls) == len(snap.controls)

    def test_load_returns_none_when_empty(self, canonical_db) -> None:
        from tract.export.canonical import ensure_export_history_table, load_prior_snapshot

        ensure_export_history_table(canonical_db)
        prior = load_prior_snapshot(canonical_db, "fw1")
        assert prior is None

    def test_content_hash_verified_on_load(self, canonical_db) -> None:
        from tract.export.canonical import (
            build_snapshot,
            ensure_export_history_table,
            load_prior_snapshot,
            save_to_export_history,
        )
        from tract.crosswalk.schema import get_connection

        ensure_export_history_table(canonical_db)
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
        save_to_export_history(canonical_db, snap)

        conn = get_connection(canonical_db)
        try:
            conn.execute(
                "UPDATE export_history SET content_hash = 'corrupted' WHERE framework_id = 'fw1'"
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ValueError, match="content_hash mismatch"):
            load_prior_snapshot(canonical_db, "fw1")
