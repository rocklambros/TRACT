"""Tests for dataset bundle — JSONL dedup, assignment_type, framework metadata."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tract import licensing
from tract.config import PROJECT_ROOT
from tract.crosswalk.schema import create_database, get_connection
from tract.dataset.bundle import (
    _build_crosswalk_jsonl,
    _build_framework_metadata,
    _build_zenodo_metadata,
    bundle_dataset,
)
from tract.licensing import (
    LICENSE_TEXTS_DIR,
    NOTICE_FILENAME,
    PUBLISHED_LICENSE_ID,
    PUBLISHED_LICENSE_LINK,
    PUBLISHED_LICENSE_NAME,
)


@pytest.fixture()
def bundle_db(tmp_path: Path) -> Path:
    """Create a test DB with varied assignments for bundle testing."""
    db_path = tmp_path / "bundle.db"
    create_database(db_path)

    conn = get_connection(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")

    conn.execute(
        "INSERT INTO frameworks (id, name, control_count) VALUES (?, ?, ?)",
        ("fw_alpha", "Alpha Framework", 3),
    )
    conn.execute(
        "INSERT INTO frameworks (id, name, control_count) VALUES (?, ?, ?)",
        ("fw_beta", "Beta Framework", 2),
    )

    conn.execute(
        "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
        ("hub-1", "Hub One", "/security/hub-one"),
    )
    conn.execute(
        "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
        ("hub-2", "Hub Two", "/security/hub-two"),
    )
    conn.execute(
        "INSERT INTO hubs (id, name, path) VALUES (?, ?, ?)",
        ("hub-3", "Hub Three", "/security/hub-three"),
    )

    for i, fw in enumerate(["fw_alpha"] * 3 + ["fw_beta"] * 2):
        conn.execute(
            "INSERT INTO controls (id, framework_id, section_id, title) VALUES (?, ?, ?, ?)",
            (f"ctrl-{i+1}", fw, f"SEC-{i+1}", f"Control {i+1}"),
        )

    # GT linked assignment
    conn.execute(
        "INSERT INTO assignments "
        "(control_id, hub_id, confidence, provenance, source_link_id, review_status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ctrl-1", "hub-1", 1.0, "opencre_ground_truth", "LinkedTo", "ground_truth"),
    )

    # GT auto assignment
    conn.execute(
        "INSERT INTO assignments "
        "(control_id, hub_id, confidence, provenance, source_link_id, review_status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ctrl-2", "hub-2", 1.0, "opencre_ground_truth", "AutomaticallyLinkedTo", "ground_truth"),
    )

    # Duplicate: same (ctrl-1, hub-1) but lower-priority provenance
    conn.execute(
        "INSERT INTO assignments "
        "(control_id, hub_id, confidence, provenance, review_status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("ctrl-1", "hub-1", 0.85, "active_learning_round_2", "accepted"),
    )

    # Model accepted (no reassignment)
    conn.execute(
        "INSERT INTO assignments "
        "(control_id, hub_id, confidence, provenance, review_status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("ctrl-3", "hub-1", 0.75, "active_learning_round_2", "accepted"),
    )

    # Model reassigned (original_hub_id set)
    conn.execute(
        "INSERT INTO assignments "
        "(control_id, hub_id, confidence, provenance, review_status, original_hub_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ctrl-4", "hub-3", None, "model_prediction", "accepted", "hub-2"),
    )

    # Model rejected
    conn.execute(
        "INSERT INTO assignments "
        "(control_id, hub_id, confidence, provenance, review_status, reviewer_notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ctrl-5", "hub-1", 0.30, "model_prediction", "rejected", "No hub fits"),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def dummy_files(tmp_path: Path) -> dict[str, Path]:
    """Create dummy JSON files for bundle_dataset copies."""
    files = {}
    for name in ["hierarchy", "hub_descriptions", "bridge_report", "review_metrics"]:
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"type": name}), encoding="utf-8")
        files[name] = p
    return files


class TestJsonlDedup:
    def test_dedup_produces_one_row_per_pair(
        self, bundle_db: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "crosswalk.jsonl"
        count = _build_crosswalk_jsonl(bundle_db, output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        pairs = set()
        for line in lines:
            row = json.loads(line)
            pairs.add((row["control_id"], row["hub_id"]))

        assert count == len(lines)
        assert len(pairs) == len(lines)

    def test_dedup_keeps_highest_priority_provenance(
        self, bundle_db: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "crosswalk.jsonl"
        _build_crosswalk_jsonl(bundle_db, output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        ctrl1_hub1 = [
            json.loads(line) for line in lines
            if json.loads(line)["control_id"] == "ctrl-1"
            and json.loads(line)["hub_id"] == "hub-1"
        ]

        assert len(ctrl1_hub1) == 1
        assert ctrl1_hub1[0]["provenance"] == "opencre_ground_truth"


class TestAssignmentTypeDerivation:
    def test_all_five_assignment_types(
        self, bundle_db: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "crosswalk.jsonl"
        _build_crosswalk_jsonl(bundle_db, output)

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        rows = [json.loads(line) for line in lines]
        types_by_control = {r["control_id"]: r["assignment_type"] for r in rows}

        assert types_by_control["ctrl-1"] == "ground_truth_linked"
        assert types_by_control["ctrl-2"] == "ground_truth_auto"
        assert types_by_control["ctrl-3"] == "model_accepted"
        assert types_by_control["ctrl-4"] == "model_reassigned"
        assert types_by_control["ctrl-5"] == "model_rejected"


class TestJsonlRequiredFields:
    def test_rows_have_all_required_fields(
        self, bundle_db: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "crosswalk.jsonl"
        _build_crosswalk_jsonl(bundle_db, output)

        required = {
            "control_id", "framework_id", "framework_name", "section_id",
            "control_title", "hub_id", "hub_name", "hub_path",
            "assignment_type", "confidence", "provenance",
            "review_status", "reviewer_notes",
        }

        lines = output.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            row = json.loads(line)
            assert required <= set(row.keys()), f"Missing fields: {required - set(row.keys())}"


class TestFrameworkMetadata:
    def test_coverage_type_computation(
        self, bundle_db: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "fw_meta.json"
        metadata = _build_framework_metadata(bundle_db, output)

        by_id = {m["framework_id"]: m for m in metadata}

        # fw_alpha has GT + AL assignments → mixed
        assert by_id["fw_alpha"]["coverage_type"] == "mixed"

        # fw_beta has only model_prediction assignments → model_prediction
        assert by_id["fw_beta"]["coverage_type"] == "model_prediction"

    def test_framework_counts(
        self, bundle_db: Path, tmp_path: Path,
    ) -> None:
        output = tmp_path / "fw_meta.json"
        metadata = _build_framework_metadata(bundle_db, output)

        by_id = {m["framework_id"]: m for m in metadata}

        assert by_id["fw_alpha"]["total_controls"] == 3
        assert by_id["fw_alpha"]["assigned_controls"] > 0
        assert by_id["fw_beta"]["total_controls"] == 2


class TestZenodoMetadata:
    def test_zenodo_structure(self, tmp_path: Path) -> None:
        output = tmp_path / "zenodo.json"
        _build_zenodo_metadata(output)

        data = json.loads(output.read_text(encoding="utf-8"))
        assert "title" in data
        assert "description" in data
        assert "creators" in data
        assert "license" in data
        # Not a single SPDX identifier. This file asserted CC-BY-SA-4.0 over
        # content drawn from 31 publishers, and a human pastes it straight into
        # Zenodo's form, so a wrong identifier here becomes a published grant.
        assert data["license"] == PUBLISHED_LICENSE_ID
        assert data["license_name"] == PUBLISHED_LICENSE_NAME
        assert data["license_link"] == PUBLISHED_LICENSE_LINK
        assert data["license"] != "CC-BY-SA-4.0"
        assert data["resource_type"] == "dataset"
        assert "keywords" in data
        assert isinstance(data["keywords"], list)


class TestLicensingFilesShipInTheBundle:
    """The remedy has to reach the artifact, not only the source repository.

    The bundle used to write a hand-typed CC BY-SA 4.0 deed summary as LICENSE
    and ship nothing else. NOTICE carries the per-framework terms, the
    modification statement GPL-3.0 section 5(a) requires, and the open
    questions, and it was absent from the artifact most consumers touch.
    LICENSES/ carries the texts GPL-3.0 section 4 requires to travel with the
    work. Both are now copied, and these tests fail if either stops.
    """

    @pytest.fixture
    def staging(
        self, bundle_db: Path, tmp_path: Path, dummy_files: dict[str, Path],
    ) -> Path:
        target = tmp_path / "staging"
        bundle_dataset(
            db_path=bundle_db,
            staging_dir=target,
            hierarchy_path=dummy_files["hierarchy"],
            hub_descriptions_path=dummy_files["hub_descriptions"],
            bridge_report_path=dummy_files["bridge_report"],
            review_metrics_path=dummy_files["review_metrics"],
        )
        return target

    def test_the_bundle_ships_notice(self, staging: Path) -> None:
        notice = staging / NOTICE_FILENAME
        assert notice.is_file(), (
            f"{NOTICE_FILENAME} is not in the built bundle. The card's "
            f"license_link points at it, so the published artifact would link "
            f"to a file that is not there."
        )
        assert notice.read_bytes() == (PROJECT_ROOT / NOTICE_FILENAME).read_bytes()

    def test_the_shipped_notice_carries_the_modification_statement(
        self, staging: Path,
    ) -> None:
        """A NOTICE stripped of the section would satisfy existence alone."""
        body = (staging / NOTICE_FILENAME).read_text(encoding="utf-8")
        assert "Modifications to framework text" in body
        assert "DESCRIPTION_MAX_LENGTH" in body

    def test_the_bundle_ships_every_licence_text(self, staging: Path) -> None:
        shipped = staging / LICENSE_TEXTS_DIR.name
        assert shipped.is_dir(), (
            f"{LICENSE_TEXTS_DIR.name}/ is not in the built bundle. GPL-3.0 "
            f"section 4 requires the licence text to travel with the work."
        )
        names = {path.name for path in shipped.iterdir()}
        expected = {path.name for path in LICENSE_TEXTS_DIR.iterdir()}
        assert names == expected, (
            f"the bundle ships {sorted(names)} and the repository holds "
            f"{sorted(expected)}"
        )

    def test_the_bundle_license_is_the_repository_dedication(
        self, staging: Path,
    ) -> None:
        """LICENSE is TRACT's own CC0 dedication, copied rather than retyped.

        The previous body was a hand-typed summary of the CC BY-SA 4.0 deed,
        so the artifact carried a paraphrase of a licence TRACT had no standing
        to grant. Byte equality with the repository's LICENSE rules out both
        the paraphrase and any future drift.
        """
        shipped = (staging / "LICENSE").read_bytes()
        assert shipped == (PROJECT_ROOT / "LICENSE").read_bytes()
        body = shipped.decode("utf-8")
        assert "CC0 1.0 Universal" in body
        assert "CC BY-SA 4.0" not in body, (
            "the withdrawn CC BY-SA 4.0 grant is back in the bundle's LICENSE"
        )

    def test_the_bundle_refuses_to_build_without_the_licence_texts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fail loud, not quiet. A silent skip is how NOTICE went missing.

        The failure direction that matters: if LICENSES/ is ever removed, the
        publish path must stop rather than ship an artifact whose NOTICE points
        at a directory that is not there.
        """
        monkeypatch.setattr(
            licensing, "LICENSE_TEXTS_DIR", tmp_path / "absent",
        )
        with pytest.raises(FileNotFoundError, match="GPL-3.0 section 4"):
            licensing.copy_licensing_files(tmp_path / "staging_out")


class TestBundleDataset:
    def test_bundle_creates_all_files(
        self, bundle_db: Path, tmp_path: Path, dummy_files: dict[str, Path],
    ) -> None:
        staging = tmp_path / "staging"
        stats = bundle_dataset(
            db_path=bundle_db,
            staging_dir=staging,
            hierarchy_path=dummy_files["hierarchy"],
            hub_descriptions_path=dummy_files["hub_descriptions"],
            bridge_report_path=dummy_files["bridge_report"],
            review_metrics_path=dummy_files["review_metrics"],
        )

        expected_files = {
            "crosswalk_v1.0.jsonl",
            "framework_metadata.json",
            "zenodo_metadata.json",
            "cre_hierarchy_v1.1.json",
            "hub_descriptions_v1.0.json",
            "bridge_report.json",
            "review_metrics.json",
            "LICENSE",
        }
        actual_files = {p.name for p in staging.iterdir() if p.is_file()}
        assert expected_files <= actual_files

        assert stats["total_rows"] > 0
        assert stats["frameworks"] == 2
        assert len(stats["files"]) >= len(expected_files)
