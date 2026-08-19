"""Tests for canonical export snapshot builder and differ (spec §§2-7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.crosswalk.schema import create_database
from tract.export.canonical_schema import StandardSnapshot
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


class TestEmbeddingSlicer:
    def test_slices_correct_controls(self, tmp_path) -> None:
        import numpy as np
        from tract.export.canonical import slice_embeddings_for_framework

        n_controls = 10
        n_hubs = 3
        dim = 1024
        control_ids = [f"fw1::c{i}" for i in range(5)] + [f"fw2::c{i}" for i in range(5)]
        hub_ids = [f"h{i}" for i in range(n_hubs)]

        artifacts_path = tmp_path / "deployment_artifacts.npz"
        np.savez(
            str(artifacts_path),
            control_embeddings=np.random.rand(n_controls, dim).astype(np.float32),
            control_ids=np.array(control_ids),
            hub_embeddings=np.random.rand(n_hubs, dim).astype(np.float32),
            hub_ids=np.array(hub_ids),
            model_adapter_hash=np.array("abc123"),
        )

        canonical_control_ids = {"fw1:c0", "fw1:c1", "fw1:c2", "fw1:c3", "fw1:c4"}
        result = slice_embeddings_for_framework(
            artifacts_path=artifacts_path,
            canonical_control_ids=canonical_control_ids,
            model_adapter_hash="abc123",
        )
        assert result["control_ids"].shape[0] == 5
        assert result["control_embeddings"].shape == (5, dim)
        assert result["hub_embeddings"].shape == (n_hubs, dim)
        assert all("::" not in str(cid) for cid in result["control_ids"])

    def test_raises_on_hash_mismatch(self, tmp_path) -> None:
        import numpy as np
        from tract.export.canonical import slice_embeddings_for_framework

        artifacts_path = tmp_path / "deployment_artifacts.npz"
        np.savez(
            str(artifacts_path),
            control_embeddings=np.random.rand(2, 1024).astype(np.float32),
            control_ids=np.array(["fw1::c0", "fw1::c1"]),
            hub_embeddings=np.random.rand(1, 1024).astype(np.float32),
            hub_ids=np.array(["h0"]),
            model_adapter_hash=np.array("abc123"),
        )
        with pytest.raises(ValueError, match="model_adapter_hash mismatch"):
            slice_embeddings_for_framework(
                artifacts_path=artifacts_path,
                canonical_control_ids={"fw1:c0"},
                model_adapter_hash="different_hash",
            )


class TestExportCanonical:
    def test_initial_export_creates_files(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical

        output_dir = tmp_path / "output"
        result = export_canonical(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
        )
        assert (output_dir / "fw1" / "snapshot.json").exists()
        assert (output_dir / "fw1" / "changeset.json").exists()
        assert result["fw1"]["changeset_summary"]["controls_added"] > 0

    def test_second_export_detects_no_changes(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical

        output_dir = tmp_path / "output"
        kwargs = dict(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
        )
        export_canonical(**kwargs)
        result = export_canonical(**kwargs)
        summary = result["fw1"]["changeset_summary"]
        assert summary["controls_added"] == 0
        assert summary["mappings_added"] == 0

    def test_dry_run_does_not_write(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical

        output_dir = tmp_path / "output"
        result = export_canonical(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
            dry_run=True,
        )
        assert not (output_dir / "fw1" / "snapshot.json").exists()
        assert "fw1" in result

    def test_snapshot_json_validates(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical
        from tract.export.canonical_schema import StandardSnapshot

        output_dir = tmp_path / "output"
        export_canonical(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
        )
        snap_path = output_dir / "fw1" / "snapshot.json"
        snap = StandardSnapshot.model_validate_json(snap_path.read_text(encoding="utf-8"))
        assert snap.framework_id == "fw1"
        assert len(snap.content_hash) == 64


# ── Licence filtering ─────────────────────────────────────────────────────
#
# `grep -c` for RESTRICTED, licens or OVERLAY in tract/export/canonical.py and
# canonical_schema.py returned 0 for both. CanonicalControl carries full
# control text for every framework, the default output directory is
# ./canonical_export at the repository root, and that directory was not
# gitignored, so `tract export-canonical && git add -A` staged ISO 27001 and
# ETSI control statements into a CC0 repository.
#
# The gitignore line closes the git channel. These tests cover the other one:
# the command's stated destination is an OpenCRE RFC, which is a third party
# outside git, where no ignore rule applies.

_LICENSED_PROSE = (
    "Regional facility credentials shall be issued, reviewed and revoked "
    "under a documented process approved by the accountable security lead."
)


@pytest.fixture
def overlay_db(tmp_path: Path) -> tuple[Path, str]:
    """One publishable framework and one whose text may not be redistributed.

    The overlay framework is taken from the live constant rather than named
    here, so the fixture cannot drift out of step with the tier it tests.
    """
    from tract.config import OVERLAY_FRAMEWORK_IDS

    overlay_id = sorted(OVERLAY_FRAMEWORK_IDS)[0]
    db_path = tmp_path / "overlay_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "public_fw", "name": "Public FW", "version": "1.0",
         "fetch_date": "2026-05-04", "control_count": 1},
        {"id": overlay_id, "name": overlay_id.upper(), "version": "1.0",
         "fetch_date": "2026-05-04", "control_count": 1},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "public_fw:c1", "framework_id": "public_fw", "section_id": "c1",
         "title": "Public control", "description": _LICENSED_PROSE,
         "full_text": None},
        {"id": f"{overlay_id}:c1", "framework_id": overlay_id,
         "section_id": "c1", "title": "Overlay control",
         "description": _LICENSED_PROSE, "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "public_fw:c1", "hub_id": "h1", "confidence": 0.8,
         "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": None, "review_status": "accepted"},
        {"control_id": f"{overlay_id}:c1", "hub_id": "h1", "confidence": 0.8,
         "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": None, "review_status": "accepted"},
    ])
    return db_path, overlay_id


class TestExportWithholdsUnpublishableControlText:
    def _snapshot(self, db_path: Path, framework_id: str) -> StandardSnapshot:
        from tract.export.canonical import build_snapshot

        return build_snapshot(
            db_path=db_path,
            framework_id=framework_id,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name=framework_id.upper(),
        )

    def test_an_overlay_framework_exports_no_control_text(
        self, overlay_db: tuple[Path, str],
    ) -> None:
        db_path, overlay_id = overlay_db
        snap = self._snapshot(db_path, overlay_id)
        assert _LICENSED_PROSE not in snap.model_dump_json(), (
            f"{overlay_id}'s control statement reached a canonical export. "
            f"The export's destination is an OpenCRE RFC, outside git, so no "
            f".gitignore rule stops it."
        )

    def test_a_publishable_framework_keeps_its_control_text(
        self, overlay_db: tuple[Path, str],
    ) -> None:
        """The other direction. A filter that withheld everything would pass
        the test above and destroy the deliverable."""
        db_path, _ = overlay_db
        snap = self._snapshot(db_path, "public_fw")
        assert snap.controls[0].description == _LICENSED_PROSE

    def test_identifiers_titles_and_mappings_survive(
        self, overlay_db: tuple[Path, str],
    ) -> None:
        """Withheld text, not a withheld framework.

        Omitting overlay frameworks would drop their CRE mappings from the
        proposal, which withholds TRACT's own CC0 contribution in order to
        protect somebody else's text. OpenCRE already publishes these section
        identifiers and names.
        """
        db_path, overlay_id = overlay_db
        snap = self._snapshot(db_path, overlay_id)
        assert len(snap.controls) == 1
        assert len(snap.mappings) == 1
        control = snap.controls[0]
        assert control.section_id == "c1"
        assert control.title == "Overlay control"
        assert control.hyperlink
        assert snap.mappings[0].hub_id == "h1"

    def test_the_placeholder_says_why_and_names_the_licence(
        self, overlay_db: tuple[Path, str],
    ) -> None:
        """An empty string reads as "no description", which is a lie.

        A recipient holding only snapshot.json has to be able to tell a
        withheld statement from an absent one.
        """
        from tract.config import FRAMEWORK_LICENSES
        from tract.licensing import NOTICE_FILENAME, spdx_identifiers

        db_path, overlay_id = overlay_db
        description = self._snapshot(db_path, overlay_id).controls[0].description
        assert description, "the placeholder is empty, so it explains nothing"
        assert "withheld" in description.lower()
        assert overlay_id in description
        assert NOTICE_FILENAME in description
        identifiers = spdx_identifiers(FRAMEWORK_LICENSES.get(overlay_id, ""))
        for identifier in identifiers:
            assert identifier in description

    def test_every_overlay_framework_is_filtered_not_just_the_first(self) -> None:
        """Driven by the set. A per-framework hole would survive the tests above."""
        from tract.config import OVERLAY_FRAMEWORK_IDS
        from tract.export.canonical import exportable_description

        assert OVERLAY_FRAMEWORK_IDS, "no overlay tier to enforce"
        leaked = sorted(
            framework_id for framework_id in OVERLAY_FRAMEWORK_IDS
            if exportable_description(framework_id, _LICENSED_PROSE)
            == _LICENSED_PROSE
        )
        assert not leaked, f"{leaked} export their control text unfiltered"

    def test_a_control_with_no_framework_raises(self) -> None:
        """Fail loud. Defaulting an unattributed row to publishable is how an
        unfiltered source would reach an RFC."""
        from tract.export.canonical import exportable_description

        with pytest.raises(ValueError, match="no framework_id"):
            exportable_description("", _LICENSED_PROSE)

    def test_the_written_snapshot_and_changeset_carry_no_control_text(
        self, overlay_db: tuple[Path, str], tmp_path: Path,
    ) -> None:
        """End to end, on the bytes that actually leave the machine."""
        from tract.export.canonical import export_canonical

        db_path, overlay_id = overlay_db
        output_dir = tmp_path / "canonical_export"
        results = export_canonical(
            db_path=db_path,
            framework_ids=[overlay_id, "public_fw"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={overlay_id: overlay_id.upper(), "public_fw": "Public FW"},
        )
        assert results[overlay_id]["control_text_withheld"] is True
        assert results["public_fw"]["control_text_withheld"] is False

        for name in ("snapshot.json", "changeset.json"):
            body = (output_dir / overlay_id / name).read_text(encoding="utf-8")
            assert _LICENSED_PROSE not in body, (
                f"{overlay_id}/{name} carries the publisher's control text"
            )
        public_body = (
            output_dir / "public_fw" / "snapshot.json"
        ).read_text(encoding="utf-8")
        assert _LICENSED_PROSE in public_body


def test_the_default_export_directory_is_gitignored() -> None:
    """The smaller half of the fix, and the one a stray `git add -A` needs.

    Asserted against git rather than against the .gitignore text, because a
    line that is present and shadowed by a later negation ignores nothing.
    """
    import subprocess
    from pathlib import Path

    from tract.config import PHASE5_CANONICAL_EXPORT_DIR

    repo_root = Path(__file__).resolve().parent.parent
    probe = (
        PHASE5_CANONICAL_EXPORT_DIR.relative_to(repo_root) / "fw" / "snapshot.json"
    )
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(probe)],
        cwd=repo_root, capture_output=True,
    )
    assert result.returncode == 0, (
        f"{probe} is not ignored by git. `tract export-canonical && "
        f"git add -A` would stage every framework's control text."
    )
